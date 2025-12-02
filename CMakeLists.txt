cmake_minimum_required(VERSION 3.10)
project(splat-transform)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LibZip REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(WebP REQUIRED)

add_library(splat_trans STATIC)

add_subdirectory(utils)
add_subdirectory(maths)
add_subdirectory(serialize)
add_subdirectory(readers)
add_subdirectory(writers)

target_link_libraries(splat_trans Eigen3::Eigen ${WebP_LIBRARIES} libzip::zip)
target_include_directories(splat_trans PUBLIC ${WebP_INCLUDE_DIRS})
if(WIN32)
    target_compile_definitions(splat_trans PUBLIC _USE_MATH_DEFINES)
endif()

add_subdirectory(app)