cmake_minimum_required(VERSION 3.10...3.26)

# Platform detection for GPU support
set(APPLE_SILICON FALSE)
if(APPLE OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(APPLE_SILICON TRUE)
endif()

# GPU backend options
if(APPLE_SILICON)
    set(_DEFAULT_CUDA OFF)
    set(_DEFAULT_METAL ON)
else()
    set(_DEFAULT_CUDA ON)
    set(_DEFAULT_METAL OFF)
endif()

option(SPLAT_USE_CUDA "Use CUDA for GPU acceleration" ${_DEFAULT_CUDA})
option(SPLAT_USE_METAL "Use Metal for GPU acceleration (macOS only)" ${_DEFAULT_METAL})

# Set CUDA architectures only if CUDA is enabled
if(SPLAT_USE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;89" CACHE STRING "CUDA architectures")
endif()

# Configure project languages based on GPU backend
set(_PROJECT_LANGUAGES CXX C)
if(SPLAT_USE_CUDA)
    list(APPEND _PROJECT_LANGUAGES CUDA)
endif()

project(SplatLib 
    VERSION 1.3.1 
    LANGUAGES ${_PROJECT_LANGUAGES}
    DESCRIPTION "Splat Transform")
set(SPLAT_INFO "${PROJECT_NAME} ${PROJECT_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(SPLAT_USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

if(SPLAT_USE_METAL)
    enable_language(OBJC)
    enable_language(OBJCXX)
    set(CMAKE_OBJC_STANDARD 17)
    set(CMAKE_OBJC_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()

option(BUILD_SPLAT_SHARED "Build splat as a shared library" ON)
option(BUILD_SPLAT_CAPI "Build C API" ON)
option(BUILD_SPLAT_VIEWER "Build ImGui + VTK splat viewer" OFF)
option(BUILD_SPLAT_TRANSFORM_TOOL "Build splat file format transform tool" OFF)
option(BUILD_SPLAT_TESTS "Build unit tests" OFF)
option(BUILD_SPLAT_VISUALIZATION "Build optional VTK visualization helpers" OFF)

option(BUILD_SPLAT_PLAYCANVAS_LOD "Build PlayCanvas LOD generation tool" OFF)
option(BUILD_SPLAT_PLAYCANVAS_VIEWER "Build PlayCanvas-style OpenGL splat viewer" OFF)

if(BUILD_SPLAT_VIEWER AND NOT BUILD_SPLAT_VISUALIZATION)
    set(BUILD_SPLAT_VISUALIZATION ON CACHE BOOL "Build optional VTK visualization helpers" FORCE)
endif()

if(BUILD_SPLAT_VISUALIZATION)
    set(SPLAT_VTK_COMPONENTS
        CommonCore
        CommonDataModel
        InteractionStyle
        InteractionWidgets
        RenderingAnnotation
        RenderingCore
        RenderingOpenGL2
    )
endif()

find_package(CUDAToolkit)
if(SPLAT_USE_CUDA AND NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA requested but CUDAToolkit not found")
endif()

if(SPLAT_USE_METAL AND NOT APPLE)
    message(FATAL_ERROR "Metal is only supported on macOS")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(zstd CONFIG REQUIRED)
find_package(WebP REQUIRED)
if(BUILD_SPLAT_VISUALIZATION)
    find_package(VTK REQUIRED COMPONENTS ${SPLAT_VTK_COMPONENTS})
endif()
find_package(Doxygen)

set(MESHOPT_BUILD_DEMO OFF)
set(MESHOPT_BUILD_GLTFPACK OFF)
set(MESHOPT_INSTALL OFF)

if(BUILD_SPLAT_SHARED)
    set(MESHOPT_BUILD_SHARED_LIBS ON CACHE BOOL "Build meshoptimizer as shared library" FORCE)
endif()
add_subdirectory(thirdparty/meshoptimizer)

# Export GPU backend configuration to subdirectories
set(SPLAT_GPU_BACKEND "NONE")
if(SPLAT_USE_CUDA)
    set(SPLAT_GPU_BACKEND "CUDA")
elseif(SPLAT_USE_METAL)
    set(SPLAT_GPU_BACKEND "METAL")
endif()

message(STATUS "SplatLib GPU Backend: ${SPLAT_GPU_BACKEND}")

add_subdirectory(src)

if(BUILD_SPLAT_CAPI)
    add_subdirectory(capi)
endif()

if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxygen.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
    
    message(STATUS "Doxygen found: Target 'doc' is available.")
else()
    message(STATUS "Doxygen not found: Target 'doc' will not be created.")
endif()

if(BUILD_SPLAT_TRANSFORM_TOOL)
    add_subdirectory(transform)
endif()

add_subdirectory(playcanvas)

if(BUILD_SPLAT_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_SPLAT_VIEWER)
    add_subdirectory(viewer)
endif()
