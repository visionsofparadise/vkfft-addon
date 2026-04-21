#include <napi.h>

#include "vulkan/vulkan.h"
#include "glslang_c_interface.h"
#include "vkFFT.h"

#include <unordered_map>
#include <mutex>
#include <cstring>
#include <string>
#include <cstdint>

// ── Vulkan GPU context ──────────────────────────────────────────────────────
struct VkGPUContext {
	VkInstance instance = VK_NULL_HANDLE;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkPhysicalDeviceProperties physicalDeviceProperties = {};
	VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties = {};
	VkDevice device = VK_NULL_HANDLE;
	uint32_t queueFamilyIndex = 0;
	VkQueue queue = VK_NULL_HANDLE;
	VkCommandPool commandPool = VK_NULL_HANDLE;
	VkFence fence = VK_NULL_HANDLE;
};

static VkGPUContext gpu = {};
static bool gpu_initialized = false;
static bool gpu_available = false;
static std::mutex gpu_mutex;

// ── Cached VkFFT plans ──────────────────────────────────────────────────────
// Cache key scheme (for when caching is wired up — currently plans are built
// and destroyed per call, matching the existing 1D code path):
//   1D: (fftDim=1, rows=1, cols=fftSize, batchCount)
//   2D: (fftDim=2, rows,   cols,         batchCount)
// shape_key = (uint64_t)fftDim << 56 | (uint64_t)(uint16_t)rows << 40
//           | (uint64_t)(uint16_t)cols << 24 | (uint64_t)(uint32_t)batchCount
// This keeps 1D and 2D plans in disjoint key ranges.
struct CachedPlan {
	VkFFTApplication app;
	uint64_t bufferSize;
	int fftDim;       // 1 or 2
	int rows;         // 1 for 1D
	int cols;         // fftSize for 1D
	int batchCount;
};

static std::unordered_map<uint64_t, CachedPlan*> cached_plans;
static std::mutex plan_mutex;

// ── Vulkan helpers ──────────────────────────────────────────────────────────

static VkFFTResult findMemoryType(uint64_t memoryTypeBits, uint64_t memorySize, VkMemoryPropertyFlags properties, uint32_t* memoryTypeIndex) {
	for (uint32_t i = 0; i < gpu.physicalDeviceMemoryProperties.memoryTypeCount; i++) {
		if ((memoryTypeBits & ((uint64_t)1 << i)) &&
			((gpu.physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties) &&
			(gpu.physicalDeviceMemoryProperties.memoryHeaps[gpu.physicalDeviceMemoryProperties.memoryTypes[i].heapIndex].size >= memorySize)) {
			*memoryTypeIndex = i;
			return VKFFT_SUCCESS;
		}
	}
	return VKFFT_ERROR_FAILED_TO_FIND_MEMORY;
}

static VkFFTResult allocateBuffer(VkBuffer* buffer, VkDeviceMemory* deviceMemory, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, uint64_t size) {
	VkBufferCreateInfo bufferCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	bufferCreateInfo.size = size;
	bufferCreateInfo.usage = usageFlags;
	VkResult res = vkCreateBuffer(gpu.device, &bufferCreateInfo, nullptr, buffer);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_CREATE_BUFFER;

	VkMemoryRequirements memReq = {};
	vkGetBufferMemoryRequirements(gpu.device, *buffer, &memReq);

	VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
	allocInfo.allocationSize = memReq.size;
	VkFFTResult resFFT = findMemoryType(memReq.memoryTypeBits, memReq.size, propertyFlags, &allocInfo.memoryTypeIndex);
	if (resFFT != VKFFT_SUCCESS) return resFFT;

	res = vkAllocateMemory(gpu.device, &allocInfo, nullptr, deviceMemory);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE_MEMORY;

	res = vkBindBufferMemory(gpu.device, *buffer, *deviceMemory, 0);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_BIND_BUFFER_MEMORY;

	return VKFFT_SUCCESS;
}

static VkFFTResult transferDataFromCPU(void* cpu_arr, VkBuffer* buffer, uint64_t transferSize) {
	VkBuffer stagingBuffer = VK_NULL_HANDLE;
	VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
	VkFFTResult resFFT = allocateBuffer(&stagingBuffer, &stagingMemory,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, transferSize);
	if (resFFT != VKFFT_SUCCESS) return resFFT;

	void* data;
	vkMapMemory(gpu.device, stagingMemory, 0, transferSize, 0, &data);
	memcpy(data, cpu_arr, transferSize);
	vkUnmapMemory(gpu.device, stagingMemory);

	VkCommandBufferAllocateInfo cmdAllocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	cmdAllocInfo.commandPool = gpu.commandPool;
	cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdAllocInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	vkAllocateCommandBuffers(gpu.device, &cmdAllocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	VkBufferCopy copyRegion = {};
	copyRegion.size = transferSize;
	vkCmdCopyBuffer(commandBuffer, stagingBuffer, *buffer, 1, &copyRegion);

	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	vkQueueSubmit(gpu.queue, 1, &submitInfo, gpu.fence);
	vkWaitForFences(gpu.device, 1, &gpu.fence, VK_TRUE, 100000000000);
	vkResetFences(gpu.device, 1, &gpu.fence);
	vkFreeCommandBuffers(gpu.device, gpu.commandPool, 1, &commandBuffer);

	vkDestroyBuffer(gpu.device, stagingBuffer, nullptr);
	vkFreeMemory(gpu.device, stagingMemory, nullptr);

	return VKFFT_SUCCESS;
}

static VkFFTResult transferDataToCPU(void* cpu_arr, VkBuffer* buffer, uint64_t transferSize) {
	VkBuffer stagingBuffer = VK_NULL_HANDLE;
	VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
	VkFFTResult resFFT = allocateBuffer(&stagingBuffer, &stagingMemory,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, transferSize);
	if (resFFT != VKFFT_SUCCESS) return resFFT;

	VkCommandBufferAllocateInfo cmdAllocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	cmdAllocInfo.commandPool = gpu.commandPool;
	cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdAllocInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	vkAllocateCommandBuffers(gpu.device, &cmdAllocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	VkBufferCopy copyRegion = {};
	copyRegion.size = transferSize;
	vkCmdCopyBuffer(commandBuffer, *buffer, stagingBuffer, 1, &copyRegion);

	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	vkQueueSubmit(gpu.queue, 1, &submitInfo, gpu.fence);
	vkWaitForFences(gpu.device, 1, &gpu.fence, VK_TRUE, 100000000000);
	vkResetFences(gpu.device, 1, &gpu.fence);
	vkFreeCommandBuffers(gpu.device, gpu.commandPool, 1, &commandBuffer);

	void* data;
	vkMapMemory(gpu.device, stagingMemory, 0, transferSize, 0, &data);
	memcpy(cpu_arr, data, transferSize);
	vkUnmapMemory(gpu.device, stagingMemory);

	vkDestroyBuffer(gpu.device, stagingBuffer, nullptr);
	vkFreeMemory(gpu.device, stagingMemory, nullptr);

	return VKFFT_SUCCESS;
}

static VkFFTResult executeVkFFT(VkFFTApplication* app, int inverse) {
	VkCommandBufferAllocateInfo cmdAllocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
	cmdAllocInfo.commandPool = gpu.commandPool;
	cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdAllocInfo.commandBufferCount = 1;
	VkCommandBuffer commandBuffer = {};
	VkResult res = vkAllocateCommandBuffers(gpu.device, &cmdAllocInfo, &commandBuffer);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;

	VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	res = vkBeginCommandBuffer(commandBuffer, &beginInfo);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;

	VkFFTLaunchParams launchParams = {};
	launchParams.commandBuffer = &commandBuffer;
	VkFFTResult resFFT = VkFFTAppend(app, inverse, &launchParams);
	if (resFFT != VKFFT_SUCCESS) return resFFT;

	res = vkEndCommandBuffer(commandBuffer);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;

	VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;
	res = vkQueueSubmit(gpu.queue, 1, &submitInfo, gpu.fence);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;

	res = vkWaitForFences(gpu.device, 1, &gpu.fence, VK_TRUE, 100000000000);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;

	res = vkResetFences(gpu.device, 1, &gpu.fence);
	if (res != VK_SUCCESS) return VKFFT_ERROR_FAILED_TO_RESET_FENCES;

	vkFreeCommandBuffers(gpu.device, gpu.commandPool, 1, &commandBuffer);

	return VKFFT_SUCCESS;
}

// ── GPU initialization ──────────────────────────────────────────────────────

static bool initGPU() {
	std::lock_guard<std::mutex> lock(gpu_mutex);
	if (gpu_initialized) return gpu_available;
	gpu_initialized = true;

	// Create Vulkan instance
	VkApplicationInfo appInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
	appInfo.pApplicationName = "VkFFTAddon";
	appInfo.applicationVersion = 1;
	appInfo.pEngineName = "VkFFT";
	appInfo.engineVersion = 1;
	appInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	createInfo.pApplicationInfo = &appInfo;
	createInfo.enabledLayerCount = 0;
	createInfo.enabledExtensionCount = 0;
	createInfo.pNext = nullptr;

	if (vkCreateInstance(&createInfo, nullptr, &gpu.instance) != VK_SUCCESS) return false;

	// Find physical device (use first available)
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(gpu.instance, &deviceCount, nullptr);
	if (deviceCount == 0) {
		vkDestroyInstance(gpu.instance, nullptr);
		gpu.instance = VK_NULL_HANDLE;
		return false;
	}

	VkPhysicalDevice* devices = new VkPhysicalDevice[deviceCount];
	vkEnumeratePhysicalDevices(gpu.instance, &deviceCount, devices);
	gpu.physicalDevice = devices[0];
	delete[] devices;

	vkGetPhysicalDeviceProperties(gpu.physicalDevice, &gpu.physicalDeviceProperties);
	vkGetPhysicalDeviceMemoryProperties(gpu.physicalDevice, &gpu.physicalDeviceMemoryProperties);

	// Find compute queue family
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(gpu.physicalDevice, &queueFamilyCount, nullptr);
	VkQueueFamilyProperties* queueFamilies = new VkQueueFamilyProperties[queueFamilyCount];
	vkGetPhysicalDeviceQueueFamilyProperties(gpu.physicalDevice, &queueFamilyCount, queueFamilies);

	bool foundQueue = false;
	for (uint32_t i = 0; i < queueFamilyCount; i++) {
		if (queueFamilies[i].queueCount > 0 && (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
			gpu.queueFamilyIndex = i;
			foundQueue = true;
			break;
		}
	}
	delete[] queueFamilies;

	if (!foundQueue) {
		vkDestroyInstance(gpu.instance, nullptr);
		gpu.instance = VK_NULL_HANDLE;
		return false;
	}

	// Create logical device
	VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	queueCreateInfo.queueFamilyIndex = gpu.queueFamilyIndex;
	queueCreateInfo.queueCount = 1;
	float queuePriority = 1.0f;
	queueCreateInfo.pQueuePriorities = &queuePriority;

	VkPhysicalDeviceFeatures deviceFeatures = {};
	VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
	deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
	deviceCreateInfo.enabledExtensionCount = 0;

	if (vkCreateDevice(gpu.physicalDevice, &deviceCreateInfo, nullptr, &gpu.device) != VK_SUCCESS) {
		vkDestroyInstance(gpu.instance, nullptr);
		gpu.instance = VK_NULL_HANDLE;
		return false;
	}

	vkGetDeviceQueue(gpu.device, gpu.queueFamilyIndex, 0, &gpu.queue);

	// Create fence
	VkFenceCreateInfo fenceInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	if (vkCreateFence(gpu.device, &fenceInfo, nullptr, &gpu.fence) != VK_SUCCESS) {
		vkDestroyDevice(gpu.device, nullptr);
		vkDestroyInstance(gpu.instance, nullptr);
		gpu.device = VK_NULL_HANDLE;
		gpu.instance = VK_NULL_HANDLE;
		return false;
	}

	// Create command pool
	VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
	poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	poolInfo.queueFamilyIndex = gpu.queueFamilyIndex;
	if (vkCreateCommandPool(gpu.device, &poolInfo, nullptr, &gpu.commandPool) != VK_SUCCESS) {
		vkDestroyFence(gpu.device, gpu.fence, nullptr);
		vkDestroyDevice(gpu.device, nullptr);
		vkDestroyInstance(gpu.instance, nullptr);
		gpu.device = VK_NULL_HANDLE;
		gpu.instance = VK_NULL_HANDLE;
		return false;
	}

	gpu_available = true;
	return true;
}

// ── detectDevice() => string | null ─────────────────────────────────────────

Napi::Value DetectDevice(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();

	if (!initGPU()) {
		return env.Null();
	}

	return Napi::String::New(env, gpu.physicalDeviceProperties.deviceName);
}

// ── batchFft(input: Float32Array, fftSize: number, batchCount: number) ──────
// => { re: Float32Array, im: Float32Array }

Napi::Value BatchFft(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();

	if (info.Length() < 3) {
		Napi::TypeError::New(env, "Expected 3 arguments: input, fftSize, batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	if (!initGPU()) {
		Napi::Error::New(env, "Vulkan GPU not available").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	Napi::Float32Array input = info[0].As<Napi::Float32Array>();
	int fft_size = info[1].As<Napi::Number>().Int32Value();
	int batch_count = info[2].As<Napi::Number>().Int32Value();
	int complex_size = fft_size / 2 + 1;

	if ((int)input.ElementLength() < fft_size * batch_count) {
		Napi::RangeError::New(env, "Input buffer too small for fftSize * batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	// R2C buffer: each batch needs 2*(fftSize/2+1) floats (padded complex output)
	uint64_t bufferSizePerBatch = (uint64_t)sizeof(float) * 2 * complex_size;
	uint64_t totalBufferSize = bufferSizePerBatch * batch_count;

	// Allocate GPU buffer
	VkBuffer gpuBuffer = VK_NULL_HANDLE;
	VkDeviceMemory gpuMemory = VK_NULL_HANDLE;
	VkFFTResult resFFT = allocateBuffer(&gpuBuffer, &gpuMemory,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, totalBufferSize);
	if (resFFT != VKFFT_SUCCESS) {
		Napi::Error::New(env, "Failed to allocate GPU buffer").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	// Prepare input: pack real data into padded layout (each batch padded to 2*complex_size floats)
	float* padded_input = (float*)calloc(totalBufferSize, 1);
	float* input_data = input.Data();
	for (int b = 0; b < batch_count; b++) {
		memcpy(padded_input + b * 2 * complex_size, input_data + b * fft_size, fft_size * sizeof(float));
	}

	// Transfer to GPU
	resFFT = transferDataFromCPU(padded_input, &gpuBuffer, totalBufferSize);
	free(padded_input);
	if (resFFT != VKFFT_SUCCESS) {
		vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
		vkFreeMemory(gpu.device, gpuMemory, nullptr);
		Napi::Error::New(env, "Failed to transfer data to GPU").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	// Configure and initialize VkFFT for R2C
	VkFFTConfiguration configuration = {};
	VkFFTApplication app = {};

	configuration.FFTdim = 1;
	configuration.size[0] = fft_size;
	configuration.numberBatches = batch_count;
	configuration.performR2C = 1;
	configuration.device = &gpu.device;
	configuration.queue = &gpu.queue;
	configuration.fence = &gpu.fence;
	configuration.commandPool = &gpu.commandPool;
	configuration.physicalDevice = &gpu.physicalDevice;
	configuration.isCompilerInitialized = 0;
	configuration.buffer = &gpuBuffer;
	configuration.bufferSize = &totalBufferSize;

	resFFT = initializeVkFFT(&app, configuration);
	if (resFFT != VKFFT_SUCCESS) {
		vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
		vkFreeMemory(gpu.device, gpuMemory, nullptr);
		Napi::Error::New(env, std::string("Failed to initialize VkFFT: ") + std::to_string(resFFT)).ThrowAsJavaScriptException();
		return env.Undefined();
	}

	// Execute forward FFT (inverse=0)
	resFFT = executeVkFFT(&app, 0);
	if (resFFT != VKFFT_SUCCESS) {
		deleteVkFFT(&app);
		vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
		vkFreeMemory(gpu.device, gpuMemory, nullptr);
		Napi::Error::New(env, "Failed to execute FFT").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	// Transfer results back to CPU
	float* output_data = (float*)malloc(totalBufferSize);
	resFFT = transferDataToCPU(output_data, &gpuBuffer, totalBufferSize);

	deleteVkFFT(&app);
	vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
	vkFreeMemory(gpu.device, gpuMemory, nullptr);

	if (resFFT != VKFFT_SUCCESS) {
		free(output_data);
		Napi::Error::New(env, "Failed to transfer data from GPU").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	// Deinterleave complex output into separate re/im arrays
	Napi::Float32Array re = Napi::Float32Array::New(env, complex_size * batch_count);
	Napi::Float32Array im = Napi::Float32Array::New(env, complex_size * batch_count);

	for (int b = 0; b < batch_count; b++) {
		float* batch_out = output_data + b * 2 * complex_size;
		for (int i = 0; i < complex_size; i++) {
			re[b * complex_size + i] = batch_out[2 * i];
			im[b * complex_size + i] = batch_out[2 * i + 1];
		}
	}

	free(output_data);

	Napi::Object result = Napi::Object::New(env);
	result.Set("re", re);
	result.Set("im", im);
	return result;
}

// ── batchFft2D(input: Float32Array, rows: number, cols: number, batchCount: number) ─
// => { re: Float32Array, im: Float32Array }
// Layout (matches fftw-addon):
//   input[b * rows * cols + r * cols + c]   — block-major, row-major within each block
//   output re/im[b * rows * (cols/2+1) + r * (cols/2+1) + c]  — R2C reduction on innermost (cols) axis
// VkFFT: size[0] is the fastest-varying axis, so we set size[0]=cols, size[1]=rows to match
// the FFTW "rows × (cols/2+1)" output layout. performR2C=1 reduces size[0] (cols).

Napi::Value BatchFft2D(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();

	if (info.Length() < 4) {
		Napi::TypeError::New(env, "Expected 4 arguments: input, rows, cols, batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	if (!initGPU()) {
		Napi::Error::New(env, "Vulkan GPU not available").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	Napi::Float32Array input = info[0].As<Napi::Float32Array>();
	int rows = info[1].As<Napi::Number>().Int32Value();
	int cols = info[2].As<Napi::Number>().Int32Value();
	int batch_count = info[3].As<Napi::Number>().Int32Value();
	int complex_cols = cols / 2 + 1;

	if (rows <= 0 || cols <= 0 || batch_count <= 0) {
		Napi::RangeError::New(env, "rows, cols, batchCount must be positive").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	if ((int)input.ElementLength() < rows * cols * batch_count) {
		Napi::RangeError::New(env, "Input buffer too small for rows * cols * batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	// R2C 2D: per-batch GPU buffer is rows * 2 * complex_cols floats (padded complex output)
	uint64_t bufferSizePerBatch = (uint64_t)sizeof(float) * 2 * complex_cols * rows;
	uint64_t totalBufferSize = bufferSizePerBatch * batch_count;

	VkBuffer gpuBuffer = VK_NULL_HANDLE;
	VkDeviceMemory gpuMemory = VK_NULL_HANDLE;
	VkFFTResult resFFT = allocateBuffer(&gpuBuffer, &gpuMemory,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, totalBufferSize);
	if (resFFT != VKFFT_SUCCESS) {
		Napi::Error::New(env, "Failed to allocate GPU buffer").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	// Pack input: each row of each batch block is rows * cols reals, padded to rows * 2*complex_cols
	// In VkFFT R2C layout, the padding goes at the end of each innermost row (size[0] axis).
	float* padded_input = (float*)calloc(totalBufferSize, 1);
	float* input_data = input.Data();
	for (int b = 0; b < batch_count; b++) {
		for (int r = 0; r < rows; r++) {
			float* src = input_data + b * rows * cols + r * cols;
			float* dst = padded_input + b * 2 * complex_cols * rows + r * 2 * complex_cols;
			memcpy(dst, src, cols * sizeof(float));
		}
	}

	resFFT = transferDataFromCPU(padded_input, &gpuBuffer, totalBufferSize);
	free(padded_input);
	if (resFFT != VKFFT_SUCCESS) {
		vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
		vkFreeMemory(gpu.device, gpuMemory, nullptr);
		Napi::Error::New(env, "Failed to transfer data to GPU").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	VkFFTConfiguration configuration = {};
	VkFFTApplication app = {};

	configuration.FFTdim = 2;
	configuration.size[0] = cols;   // fastest-varying, R2C reduction axis
	configuration.size[1] = rows;
	configuration.numberBatches = batch_count;
	configuration.performR2C = 1;
	configuration.device = &gpu.device;
	configuration.queue = &gpu.queue;
	configuration.fence = &gpu.fence;
	configuration.commandPool = &gpu.commandPool;
	configuration.physicalDevice = &gpu.physicalDevice;
	configuration.isCompilerInitialized = 0;
	configuration.buffer = &gpuBuffer;
	configuration.bufferSize = &totalBufferSize;

	resFFT = initializeVkFFT(&app, configuration);
	if (resFFT != VKFFT_SUCCESS) {
		vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
		vkFreeMemory(gpu.device, gpuMemory, nullptr);
		Napi::Error::New(env, std::string("Failed to initialize VkFFT 2D: ") + std::to_string(resFFT)).ThrowAsJavaScriptException();
		return env.Undefined();
	}

	resFFT = executeVkFFT(&app, 0);
	if (resFFT != VKFFT_SUCCESS) {
		deleteVkFFT(&app);
		vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
		vkFreeMemory(gpu.device, gpuMemory, nullptr);
		Napi::Error::New(env, "Failed to execute 2D FFT").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	float* output_data = (float*)malloc(totalBufferSize);
	resFFT = transferDataToCPU(output_data, &gpuBuffer, totalBufferSize);

	deleteVkFFT(&app);
	vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
	vkFreeMemory(gpu.device, gpuMemory, nullptr);

	if (resFFT != VKFFT_SUCCESS) {
		free(output_data);
		Napi::Error::New(env, "Failed to transfer data from GPU").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	// Deinterleave: output layout per batch is rows * complex_cols complex bins (padded stride = 2*complex_cols per row)
	Napi::Float32Array re = Napi::Float32Array::New(env, rows * complex_cols * batch_count);
	Napi::Float32Array im = Napi::Float32Array::New(env, rows * complex_cols * batch_count);

	for (int b = 0; b < batch_count; b++) {
		for (int r = 0; r < rows; r++) {
			float* batch_out = output_data + b * 2 * complex_cols * rows + r * 2 * complex_cols;
			int dst_row_off = b * rows * complex_cols + r * complex_cols;
			for (int i = 0; i < complex_cols; i++) {
				re[dst_row_off + i] = batch_out[2 * i];
				im[dst_row_off + i] = batch_out[2 * i + 1];
			}
		}
	}

	free(output_data);

	Napi::Object result = Napi::Object::New(env);
	result.Set("re", re);
	result.Set("im", im);
	return result;
}

// ── batchIfft2D(re, im, rows, cols, batchCount) => Float32Array ────────────
// Inverse of batchFft2D. Scaled by 1/(rows*cols) so forward→inverse is identity.

Napi::Value BatchIfft2D(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();

	if (info.Length() < 5) {
		Napi::TypeError::New(env, "Expected 5 arguments: re, im, rows, cols, batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	if (!initGPU()) {
		Napi::Error::New(env, "Vulkan GPU not available").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	Napi::Float32Array re = info[0].As<Napi::Float32Array>();
	Napi::Float32Array im = info[1].As<Napi::Float32Array>();
	int rows = info[2].As<Napi::Number>().Int32Value();
	int cols = info[3].As<Napi::Number>().Int32Value();
	int batch_count = info[4].As<Napi::Number>().Int32Value();
	int complex_cols = cols / 2 + 1;

	if (rows <= 0 || cols <= 0 || batch_count <= 0) {
		Napi::RangeError::New(env, "rows, cols, batchCount must be positive").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	int expected_complex_len = rows * complex_cols * batch_count;
	if ((int)re.ElementLength() < expected_complex_len || (int)im.ElementLength() < expected_complex_len) {
		Napi::RangeError::New(env, "re/im buffer too small for rows * (cols/2+1) * batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	uint64_t bufferSizePerBatch = (uint64_t)sizeof(float) * 2 * complex_cols * rows;
	uint64_t totalBufferSize = bufferSizePerBatch * batch_count;

	// Interleave re/im into padded per-row complex layout (stride = 2*complex_cols per row)
	float* interleaved = (float*)malloc(totalBufferSize);
	for (int b = 0; b < batch_count; b++) {
		for (int r = 0; r < rows; r++) {
			float* batch_buf = interleaved + b * 2 * complex_cols * rows + r * 2 * complex_cols;
			int src_row_off = b * rows * complex_cols + r * complex_cols;
			for (int i = 0; i < complex_cols; i++) {
				batch_buf[2 * i] = re[src_row_off + i];
				batch_buf[2 * i + 1] = im[src_row_off + i];
			}
		}
	}

	VkBuffer gpuBuffer = VK_NULL_HANDLE;
	VkDeviceMemory gpuMemory = VK_NULL_HANDLE;
	VkFFTResult resFFT = allocateBuffer(&gpuBuffer, &gpuMemory,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, totalBufferSize);
	if (resFFT != VKFFT_SUCCESS) {
		free(interleaved);
		Napi::Error::New(env, "Failed to allocate GPU buffer").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	resFFT = transferDataFromCPU(interleaved, &gpuBuffer, totalBufferSize);
	free(interleaved);
	if (resFFT != VKFFT_SUCCESS) {
		vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
		vkFreeMemory(gpu.device, gpuMemory, nullptr);
		Napi::Error::New(env, "Failed to transfer data to GPU").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	VkFFTConfiguration configuration = {};
	VkFFTApplication app = {};

	configuration.FFTdim = 2;
	configuration.size[0] = cols;
	configuration.size[1] = rows;
	configuration.numberBatches = batch_count;
	configuration.performR2C = 1;
	configuration.device = &gpu.device;
	configuration.queue = &gpu.queue;
	configuration.fence = &gpu.fence;
	configuration.commandPool = &gpu.commandPool;
	configuration.physicalDevice = &gpu.physicalDevice;
	configuration.isCompilerInitialized = 0;
	configuration.buffer = &gpuBuffer;
	configuration.bufferSize = &totalBufferSize;

	resFFT = initializeVkFFT(&app, configuration);
	if (resFFT != VKFFT_SUCCESS) {
		vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
		vkFreeMemory(gpu.device, gpuMemory, nullptr);
		Napi::Error::New(env, std::string("Failed to initialize VkFFT 2D inverse: ") + std::to_string(resFFT)).ThrowAsJavaScriptException();
		return env.Undefined();
	}

	resFFT = executeVkFFT(&app, 1);
	if (resFFT != VKFFT_SUCCESS) {
		deleteVkFFT(&app);
		vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
		vkFreeMemory(gpu.device, gpuMemory, nullptr);
		Napi::Error::New(env, "Failed to execute 2D iFFT").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	float* output_data = (float*)malloc(totalBufferSize);
	resFFT = transferDataToCPU(output_data, &gpuBuffer, totalBufferSize);

	deleteVkFFT(&app);
	vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
	vkFreeMemory(gpu.device, gpuMemory, nullptr);

	if (resFFT != VKFFT_SUCCESS) {
		free(output_data);
		Napi::Error::New(env, "Failed to transfer data from GPU").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	// Extract real values, strip per-row padding, scale by 1/(rows*cols)
	Napi::Float32Array output = Napi::Float32Array::New(env, rows * cols * batch_count);
	float scale = 1.0f / (float)(rows * cols);

	for (int b = 0; b < batch_count; b++) {
		for (int r = 0; r < rows; r++) {
			float* batch_buf = output_data + b * 2 * complex_cols * rows + r * 2 * complex_cols;
			int dst_row_off = b * rows * cols + r * cols;
			for (int i = 0; i < cols; i++) {
				output[dst_row_off + i] = batch_buf[i] * scale;
			}
		}
	}

	free(output_data);

	return output;
}

// ── batchIfft(re: Float32Array, im: Float32Array, fftSize: number, batchCount: number) ──
// => Float32Array

Napi::Value BatchIfft(const Napi::CallbackInfo& info) {
	Napi::Env env = info.Env();

	if (info.Length() < 4) {
		Napi::TypeError::New(env, "Expected 4 arguments: re, im, fftSize, batchCount").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	if (!initGPU()) {
		Napi::Error::New(env, "Vulkan GPU not available").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	Napi::Float32Array re = info[0].As<Napi::Float32Array>();
	Napi::Float32Array im = info[1].As<Napi::Float32Array>();
	int fft_size = info[2].As<Napi::Number>().Int32Value();
	int batch_count = info[3].As<Napi::Number>().Int32Value();
	int complex_size = fft_size / 2 + 1;

	uint64_t bufferSizePerBatch = (uint64_t)sizeof(float) * 2 * complex_size;
	uint64_t totalBufferSize = bufferSizePerBatch * batch_count;

	// Interleave re/im into complex buffer
	float* interleaved = (float*)malloc(totalBufferSize);
	for (int b = 0; b < batch_count; b++) {
		float* batch_buf = interleaved + b * 2 * complex_size;
		for (int i = 0; i < complex_size; i++) {
			batch_buf[2 * i] = re[b * complex_size + i];
			batch_buf[2 * i + 1] = im[b * complex_size + i];
		}
	}

	// Allocate GPU buffer
	VkBuffer gpuBuffer = VK_NULL_HANDLE;
	VkDeviceMemory gpuMemory = VK_NULL_HANDLE;
	VkFFTResult resFFT = allocateBuffer(&gpuBuffer, &gpuMemory,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, totalBufferSize);
	if (resFFT != VKFFT_SUCCESS) {
		free(interleaved);
		Napi::Error::New(env, "Failed to allocate GPU buffer").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	// Transfer to GPU
	resFFT = transferDataFromCPU(interleaved, &gpuBuffer, totalBufferSize);
	free(interleaved);
	if (resFFT != VKFFT_SUCCESS) {
		vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
		vkFreeMemory(gpu.device, gpuMemory, nullptr);
		Napi::Error::New(env, "Failed to transfer data to GPU").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	// Configure VkFFT for C2R (same R2C config, executed with inverse=1)
	VkFFTConfiguration configuration = {};
	VkFFTApplication app = {};

	configuration.FFTdim = 1;
	configuration.size[0] = fft_size;
	configuration.numberBatches = batch_count;
	configuration.performR2C = 1;
	configuration.device = &gpu.device;
	configuration.queue = &gpu.queue;
	configuration.fence = &gpu.fence;
	configuration.commandPool = &gpu.commandPool;
	configuration.physicalDevice = &gpu.physicalDevice;
	configuration.isCompilerInitialized = 0;
	configuration.buffer = &gpuBuffer;
	configuration.bufferSize = &totalBufferSize;

	resFFT = initializeVkFFT(&app, configuration);
	if (resFFT != VKFFT_SUCCESS) {
		vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
		vkFreeMemory(gpu.device, gpuMemory, nullptr);
		Napi::Error::New(env, std::string("Failed to initialize VkFFT: ") + std::to_string(resFFT)).ThrowAsJavaScriptException();
		return env.Undefined();
	}

	// Execute inverse FFT (inverse=1 for C2R)
	resFFT = executeVkFFT(&app, 1);
	if (resFFT != VKFFT_SUCCESS) {
		deleteVkFFT(&app);
		vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
		vkFreeMemory(gpu.device, gpuMemory, nullptr);
		Napi::Error::New(env, "Failed to execute iFFT").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	// Transfer results back
	float* output_data = (float*)malloc(totalBufferSize);
	resFFT = transferDataToCPU(output_data, &gpuBuffer, totalBufferSize);

	deleteVkFFT(&app);
	vkDestroyBuffer(gpu.device, gpuBuffer, nullptr);
	vkFreeMemory(gpu.device, gpuMemory, nullptr);

	if (resFFT != VKFFT_SUCCESS) {
		free(output_data);
		Napi::Error::New(env, "Failed to transfer data from GPU").ThrowAsJavaScriptException();
		return env.Undefined();
	}

	// Extract real values from padded layout, normalize by 1/fftSize
	Napi::Float32Array output = Napi::Float32Array::New(env, fft_size * batch_count);
	float scale = 1.0f / fft_size;

	for (int b = 0; b < batch_count; b++) {
		float* batch_buf = output_data + b * 2 * complex_size;
		for (int i = 0; i < fft_size; i++) {
			output[b * fft_size + i] = batch_buf[i] * scale;
		}
	}

	free(output_data);

	return output;
}

// ── Module init ─────────────────────────────────────────────────────────────

Napi::Object Init(Napi::Env env, Napi::Object exports) {
	exports.Set("detectDevice", Napi::Function::New(env, DetectDevice));
	exports.Set("batchFft", Napi::Function::New(env, BatchFft));
	exports.Set("batchIfft", Napi::Function::New(env, BatchIfft));
	exports.Set("batchFft2D", Napi::Function::New(env, BatchFft2D));
	exports.Set("batchIfft2D", Napi::Function::New(env, BatchIfft2D));
	return exports;
}

NODE_API_MODULE(vkfft_addon, Init)
