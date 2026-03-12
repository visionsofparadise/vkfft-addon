cmake_minimum_required(VERSION 3.15)
cmake_policy(SET CMP0091 NEW)
project(vkfft_addon)

# Vulkan SDK static libs are built with /MD — match that
if(MSVC)
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" CACHE STRING "" FORCE)
endif()

add_compile_definitions(NAPI_VERSION=7)
add_compile_definitions(VKFFT_BACKEND=0)

include(FetchContent)

# VkFFT — header-only, just download the source (don't build)
FetchContent_Declare(
	vkfft
	GIT_REPOSITORY https://github.com/DTolm/VkFFT.git
	GIT_TAG v1.3.4
	GIT_SHALLOW TRUE
)
FetchContent_Populate(vkfft)

# Vulkan SDK
find_package(Vulkan REQUIRED)

# glslang + SPIRV — required by VkFFT for runtime shader compilation
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
	find_package(glslang CONFIG QUIET)
	find_package(SPIRV-Tools CONFIG QUIET)
	if(glslang_FOUND AND SPIRV-Tools_FOUND)
		# Modern glslang (14+) — use CMake targets
		set(GLSLANG_LIBS
			glslang::glslang
			glslang::SPIRV
			glslang::glslang-default-resource-limits
			SPIRV-Tools-opt
		)
		# glslang target provides parent include dir; source expects glslang_c_interface.h directly
		get_target_property(_GLSLANG_INC glslang::glslang INTERFACE_INCLUDE_DIRECTORIES)
		find_path(GLSLANG_INCLUDE glslang_c_interface.h PATHS ${_GLSLANG_INC} PATH_SUFFIXES glslang/Include)
	else()
		# Legacy layout (e.g. Ubuntu vulkan-sdk package)
		find_library(LIB_GLSLANG glslang REQUIRED)
		find_library(LIB_MACHINEINDEPENDENT MachineIndependent REQUIRED)
		find_library(LIB_GENERICCODEGEN GenericCodeGen REQUIRED)
		find_library(LIB_OSDEPENDENT OSDependent REQUIRED)
		find_library(LIB_SPIRV SPIRV REQUIRED)
		find_library(LIB_SPIRV_TOOLS SPIRV-Tools REQUIRED)
		find_library(LIB_SPIRV_TOOLS_OPT SPIRV-Tools-opt REQUIRED)
		find_library(LIB_GLSLANG_RESOURCE glslang-default-resource-limits REQUIRED)
		set(GLSLANG_LIBS
			${LIB_GLSLANG}
			${LIB_MACHINEINDEPENDENT}
			${LIB_GENERICCODEGEN}
			${LIB_OSDEPENDENT}
			${LIB_SPIRV}
			${LIB_SPIRV_TOOLS}
			${LIB_SPIRV_TOOLS_OPT}
			${LIB_GLSLANG_RESOURCE}
		)
		find_path(GLSLANG_INCLUDE glslang_c_interface.h PATH_SUFFIXES glslang/Include)
	endif()
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
