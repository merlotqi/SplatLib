file(GLOB_RECURSE io_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/io/*.cpp ${CMAKE_CURRENT_LIST_DIR}/io/*.cu)
file(GLOB_RECURSE maths_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/maths/*.cpp ${CMAKE_CURRENT_LIST_DIR}/maths/*.cu)
file(GLOB_RECURSE models_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/models/*.cpp ${CMAKE_CURRENT_LIST_DIR}/models/*.cu)
file(GLOB_RECURSE op_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/op/*.cpp ${CMAKE_CURRENT_LIST_DIR}/op/*.cu)
file(GLOB_RECURSE spatial_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/spatial/*.cpp ${CMAKE_CURRENT_LIST_DIR}/spatial/*.cu)
file(GLOB_RECURSE utils_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/utils/*.cpp ${CMAKE_CURRENT_LIST_DIR}/utils/*.cu)
file(GLOB_RECURSE voxel_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/voxel/*.cpp ${CMAKE_CURRENT_LIST_DIR}/voxel/*.cu)

if(BUILD_SPLAT_VISUALIZATION)
    file(GLOB_RECURSE visualization_sources CONFIGURE_DEPEND ${CMAKE_CURRENT_LIST_DIR}/visualization/*.cpp ${CMAKE_CURRENT_LIST_DIR}/visualization/*.cu)
    set(_sources ${io_sources} ${maths_sources} ${models_sources} ${op_sources} ${spatial_sources} ${utils_sources} ${voxel_sources} ${visualization_sources})
else()
    set(_sources ${io_sources} ${maths_sources} ${models_sources} ${op_sources} ${spatial_sources} ${utils_sources} ${voxel_sources})
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(BUILD_SPLAT_SHARED)
    add_library(splat SHARED ${_sources} ${_cuda_sources})
else()
    add_library(splat STATIC ${_sources} ${_cuda_sources})
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
        CUDA::cudart
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        WebP::webp
        ZLIB::ZLIB
        absl::base
        absl::strings
    PRIVATE
        meshoptimizer
)

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
