cmake_minimum_required(VERSION 3.10...3.26)
set(CMAKE_CUDA_ARCHITECTURES "75;80;89" CACHE STRING "CUDA architectures")

project(SplatLib 
    VERSION 1.3.0 
    LANGUAGES CXX CUDA
    DESCRIPTION "Splat Transform")
set(SPLAT_INFO "${PROJECT_NAME} ${PROJECT_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

option(BUILD_SPLAT_SHARED "Build splat as a shared library" ON)
option(BUILD_SPLAT_CAPI "Build C API" ON)
option(BUILD_SPLAT_PYTON_BINDING "Build Python bindings" OFF)
option(BUILD_SPLAT_VIEWER "Build ImGui + OpenGL splat viewer" OFF)
option(BUILD_SPLAT_TRANSFORM_TOOL "Build splat file format transform tool" OFF)
option(BUILD_SPLAT_TESTS "Build unit tests" OFF)

find_package(Doxygen)

if(UNIX)

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(WEBP REQUIRED libwebp)
pkg_check_modules(ABSL REQUIRED absl_base absl_strings absl_any absl_hash)
pkg_check_modules(ZLIB REQUIRED zlib)
pkg_check_modules(EIGEN3 REQUIRED eigen3)
pkg_check_modules(JSON REQUIRED nlohmann_json)

set(ALL_DEPS_LIBS 
        ${EIGEN3_LIBRARIES} 
        ${JSON_LIBRARIES} 
        ${WEBP_LIBRARIES} 
        ${ZLIB_LIBRARIES} 
        ${ABSL_LIBRARIES}
        CUDA::cudart
    )
set(ALL_DEPS_INCS 
        ${EIGEN3_INCLUDE_DIRS} 
        ${JSON_INCLUDE_DIRS} 
        ${WEBP_INCLUDE_DIRS} 
        ${ZLIB_INCLUDE_DIRS} 
        ${ABSL_INCLUDE_DIRS}
)

elseif(WIN32)

find_package(CUDAToolkit REQUIRED)
find_package(WebP REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(absl REQUIRED)
find_package(ZLIB REQUIRED)
set(ALL_DEPS_LIBS 
        Eigen3::Eigen 
        nlohmann_json::nlohmann_json 
        WebP::webp 
        ZLIB::ZLIB 
        absl::base 
        absl::strings
        CUDA::cudart
    )
    set(ALL_DEPS_INCS "")

endif()

set(MESHOPT_BUILD_DEMO OFF)
set(MESHOPT_BUILD_GLTFPACK OFF)
set(MESHOPT_INSTALL OFF)

if(BUILD_SPLAT_SHARED)
    set(MESHOPT_BUILD_SHARED_LIBS ON CACHE BOOL "Build meshoptimizer as shared library" FORCE)
endif()
add_subdirectory(thirdparty/meshoptimizer)

add_subdirectory(src)

if(BUILD_SPLAT_CAPI)
    add_subdirectory(capi)
endif()

if(BUILD_SPLAT_PYTON_BINDING)
    add_subdirectory(thirdparty/pybind11)
    add_subdirectory(python)
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

if(BUILD_SPLAT_VIEWER)
    add_subdirectory(viewer)
endif()
