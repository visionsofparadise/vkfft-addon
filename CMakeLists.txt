cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
project(vkfft_addon)

add_compile_definitions(NAPI_VERSION=7)
add_compile_definitions(VKFFT_BACKEND=0)

include(FetchContent)

# VkFFT — header-only, download from GitHub
FetchContent_Declare(
	vkfft
	GIT_REPOSITORY https://github.com/DTolm/VkFFT.git
	GIT_TAG v1.3.4
	GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(vkfft)

# Vulkan SDK
find_package(Vulkan REQUIRED)

# glslang — required by VkFFT for shader compilation
# On Windows with VS 2022, link Vulkan SDK static libs directly.
# On other platforms, find via pkg-config or Vulkan SDK.
if(MSVC)
	set(VULKAN_SDK "$ENV{VULKAN_SDK}")
	set(GLSLANG_LIBS
		${VULKAN_SDK}/Lib/glslang.lib
		${VULKAN_SDK}/Lib/MachineIndependent.lib
		${VULKAN_SDK}/Lib/GenericCodeGen.lib
		${VULKAN_SDK}/Lib/OSDependent.lib
		${VULKAN_SDK}/Lib/SPIRV.lib
		${VULKAN_SDK}/Lib/SPIRV-Tools.lib
		${VULKAN_SDK}/Lib/SPIRV-Tools-opt.lib
		${VULKAN_SDK}/Lib/glslang-default-resource-limits.lib
	)
	set(GLSLANG_INCLUDE ${VULKAN_SDK}/Include/glslang/Include)
else()
	# VkFFT's FetchContent already fetches and builds glslang — use those CMake targets
	set(GLSLANG_LIBS glslang MachineIndependent GenericCodeGen OSDependent SPIRV glslang-default-resource-limits)
	# VkFFT fetches glslang into its source tree — add the Include subdir for glslang_c_interface.h
	set(GLSLANG_INCLUDE ${vkfft_SOURCE_DIR}/glslang-main/glslang/Include)
endif()

# Build the addon
add_library(${PROJECT_NAME} SHARED vkfft_addon.cpp ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

target_include_directories(${PROJECT_NAME} PRIVATE
	${CMAKE_JS_INC}
	${vkfft_SOURCE_DIR}/vkFFT
	${Vulkan_INCLUDE_DIRS}
	${GLSLANG_INCLUDE}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
	${CMAKE_JS_LIB}
	Vulkan::Vulkan
	${GLSLANG_LIBS}
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Windows: generate node.lib from .def
if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
	execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF}
		/out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()
