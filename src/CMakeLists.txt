file(GLOB_RECURSE io_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/io/*.cpp)
file(GLOB_RECURSE maths_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/maths/*.cpp)
file(GLOB_RECURSE models_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/models/*.cpp)
file(GLOB_RECURSE op_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/op/*.cpp)
file(GLOB_RECURSE spatial_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/spatial/*.cpp)
file(GLOB_RECURSE utils_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/utils/*.cpp)
file(GLOB_RECURSE voxel_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/voxel/*.cpp)

# Remove GPU backend implementations from spatial_sources - they'll be added selectively
list(FILTER spatial_sources EXCLUDE REGEX "kmeans_cpu\\.cpp$")
list(FILTER spatial_sources EXCLUDE REGEX "kmeans_metal\\.mm$")
list(FILTER spatial_sources EXCLUDE REGEX "kmeans\\.cpp$")

# GPU-specific sources
set(_gpu_sources "")
if(SPLAT_USE_CUDA)
    file(GLOB_RECURSE io_cuda CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/io/*.cu)
    file(GLOB_RECURSE maths_cuda CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/maths/*.cu)
    file(GLOB_RECURSE models_cuda CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/models/*.cu)
    file(GLOB_RECURSE op_cuda CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/op/*.cu)
    file(GLOB_RECURSE spatial_cuda CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/spatial/*.cu)
    file(GLOB_RECURSE utils_cuda CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/utils/*.cu)
    file(GLOB_RECURSE voxel_cuda CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/voxel/*.cu)
    
    # Remove kmeans_cuda from spatial_cuda since we add it explicitly
    list(FILTER spatial_cuda EXCLUDE REGEX "kmeans_cuda\\.cu$")
    list(FILTER spatial_cuda EXCLUDE REGEX "kmeans\\.cu$")
    
    set(_gpu_sources ${io_cuda} ${maths_cuda} ${models_cuda} ${op_cuda} ${spatial_cuda} ${utils_cuda} ${voxel_cuda})
    
    # Add GPU compute abstraction implementation
    list(APPEND _gpu_sources ${CMAKE_CURRENT_LIST_DIR}/spatial/kmeans_cuda.cu)
    # CUDA build keeps the original kmeans frontend in .cu.
    list(APPEND _gpu_sources ${CMAKE_CURRENT_LIST_DIR}/spatial/kmeans.cu)
    message(STATUS "SplatLib: Using CUDA compute backend")
elseif(SPLAT_USE_METAL)
    # Use macOS-specific voxelization implementation and avoid compiling CUDA-dependent source.
    list(FILTER voxel_sources EXCLUDE REGEX "gpu-voxelization\\.cpp$")
    list(FILTER voxel_sources EXCLUDE REGEX "gpu-voxelization-mac\\.cpp$")
    list(APPEND voxel_sources ${CMAKE_CURRENT_LIST_DIR}/voxel/gpu-voxelization-mac.mm)

    file(GLOB_RECURSE spatial_metal CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/spatial/*.metal)
    file(GLOB_RECURSE spatial_metal_impl CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/spatial/*_metal.mm)
    set(_gpu_sources ${spatial_metal} ${spatial_metal_impl} ${CMAKE_CURRENT_LIST_DIR}/voxel/gpu-voxelization-metal.metal)
    list(APPEND _gpu_sources ${CMAKE_CURRENT_LIST_DIR}/spatial/kmeans.cpp)
    message(STATUS "SplatLib: Using Metal compute backend")
else()
    # CPU fallback backend
    list(APPEND _gpu_sources ${CMAKE_CURRENT_LIST_DIR}/spatial/kmeans_cpu.cpp)
    list(APPEND _gpu_sources ${CMAKE_CURRENT_LIST_DIR}/spatial/kmeans.cpp)
    message(STATUS "SplatLib: Using CPU fallback compute backend")
endif()

if(BUILD_SPLAT_VISUALIZATION)
    file(GLOB_RECURSE visualization_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/visualization/*.cpp)
    set(_sources ${io_sources} ${maths_sources} ${models_sources} ${op_sources} ${spatial_sources} ${utils_sources} ${voxel_sources} ${visualization_sources} ${_gpu_sources})
else()
    set(_sources ${io_sources} ${maths_sources} ${models_sources} ${op_sources} ${spatial_sources} ${utils_sources} ${voxel_sources} ${_gpu_sources})
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CheckIncludeFileCXX)

check_include_file_cxx(nanoflann.hpp SPLAT_HAS_NANOFLANN)

if(BUILD_SPLAT_SHARED)
    add_library(splat SHARED ${_sources})
else()
    add_library(splat STATIC ${_sources})
endif()

if(BUILD_SPLAT_SHARED AND WIN32)
    set_target_properties(splat PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_library(SPLAT::splat ALIAS splat)

configure_file(
    ${PROJECT_SOURCE_DIR}/include/splat/splat_version.h.in
    ${CMAKE_BINARY_DIR}/splat/splat_version.h
)

target_include_directories(splat 
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        ${VTK_INCLUDE_DIRS}
)

target_link_libraries(splat 
    PUBLIC
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        WebP::webp
        ZLIB::ZLIB
        absl::base
        absl::strings
    PRIVATE
        meshoptimizer
)

# GPU backend specific linking
if(SPLAT_USE_CUDA)
    target_link_libraries(splat PUBLIC CUDA::cudart)
elseif(SPLAT_USE_METAL)
    target_link_libraries(splat PUBLIC 
        "-framework Metal"
        "-framework MetalKit"
        "-framework Foundation"
    )
endif()

if(BUILD_SPLAT_VISUALIZATION)
    target_link_libraries(splat
        PUBLIC
            VTK::CommonCore
            VTK::CommonDataModel
            VTK::InteractionStyle
            VTK::InteractionWidgets
            VTK::RenderingAnnotation
            VTK::RenderingCore
            VTK::RenderingOpenGL2
    )

    vtk_module_autoinit(
        TARGETS splat
        MODULES
            VTK::CommonCore
            VTK::CommonDataModel
            VTK::InteractionStyle
            VTK::InteractionWidgets
            VTK::RenderingAnnotation
            VTK::RenderingCore
            VTK::RenderingOpenGL2
    )
endif()

if(WIN32)
    target_compile_definitions(splat PUBLIC _USE_MATH_DEFINES NOMINMAX)
endif()

if(MSVC)
    target_compile_options(splat PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
endif()

if(SPLAT_HAS_NANOFLANN)
    target_compile_definitions(splat PRIVATE SPLAT_HAS_NANOFLANN=1)
    message(STATUS "nanoflann.hpp found: enabling nanoflann decimation KNN backend")
else()
    message(STATUS "nanoflann.hpp not found: using built-in decimation KNN backend")
endif()

file(GLOB_RECURSE _headers ${PROJECT_SOURCE_DIR}/include/*.h)
target_sources(splat PRIVATE ${_headers})

set(VERSION_DIR "${CMAKE_BINARY_DIR}/splat")

install(TARGETS meshoptimizer splat
    EXPORT SplatTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

install(FILES 
    ${VERSION_DIR}/splat_version.h 
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/splat
)

configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/SplatConfig.cmake.in"
    "${CMAKE_BINARY_DIR}/SplatConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/splat"
)

write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/SplatConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(EXPORT SplatTargets
    FILE SplatTargets.cmake
    NAMESPACE SPLAT::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/splat"
)

install(FILES 
    "${CMAKE_BINARY_DIR}/SplatConfig.cmake"
    "${CMAKE_BINARY_DIR}/SplatConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/splat"
)
