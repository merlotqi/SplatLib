cmake_minimum_required(VERSION 3.10)
project(splat-transform 
    VERSION 0.1.0 
    LANGUAGES CXX
    DESCRIPTION "Splat Transform")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(UNIX)
find_package(PkgConfig REQUIRED)
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
    )
set(ALL_DEPS_INCS 
        ${EIGEN3_INCLUDE_DIRS} 
        ${JSON_INCLUDE_DIRS} 
        ${WEBP_INCLUDE_DIRS} 
        ${ZLIB_INCLUDE_DIRS} 
        ${ABSL_INCLUDE_DIRS}
)

elseif(WIN32)

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
    )
    set(ALL_DEPS_INCS "")

endif()

include(GenerateExportHeader)

add_library(splat STATIC)
add_library(SPLAT::splat ALIAS splat)
add_subdirectory(include)
add_subdirectory(src)

target_link_libraries(splat 
    PUBLIC 
        ${ALL_DEPS_LIBS}
)

target_include_directories(splat 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${ALL_DEPS_INCS}
)

if(WIN32)
    target_compile_definitions(splat PUBLIC _USE_MATH_DEFINES)
endif()

add_subdirectory(app)
