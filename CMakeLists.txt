cmake_minimum_required(VERSION 3.10)
project(splat-transform 
    VERSION 0.1.0 
    LANGUAGES CXX CUDA
    DESCRIPTION "Splat Transform")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;89")
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(WEBP REQUIRED libwebp)
find_package(Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(absl REQUIRED)
find_package(ZLIB REQUIRED)
find_package(CUDAToolkit REQUIRED)

include(GenerateExportHeader)

add_library(splat STATIC)

add_subdirectory(include)
add_subdirectory(src)

target_link_libraries(splat 
    PUBLIC 
        Eigen3::Eigen 
        nlohmann_json::nlohmann_json
        CUDA::cudart
    PRIVATE 
        ${WEBP_LIBRARIES}
        absl::base 
        ZLIB::ZLIB
)

target_include_directories(splat 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${WEBP_INCLUDE_DIRS}
)

if(WIN32)
    target_compile_definitions(splat PUBLIC _USE_MATH_DEFINES)
endif()

add_subdirectory(app)
