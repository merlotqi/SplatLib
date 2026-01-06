file(GLOB_RECURSE _sources ${CMAKE_CURRENT_LIST_DIR}/*.cpp CONFIGURE_DEPEND)
file(GLOB_RECURSE _cuda_sources ${CMAKE_CURRENT_LIST_DIR}/*.cu CONFIGURE_DEPEND)

add_library(splat STATIC ${_sources} ${_cuda_sources})
add_library(SPLAT::splat ALIAS splat)

configure_file(
    ${PROJECT_SOURCE_DIR}/include/splat/splat_version.h.in
    ${CMAKE_BINARY_DIR}/splat/splat_version.h
)

target_include_directories(splat 
    PUBLIC 
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    PRIVATE   
        ${ALL_DEPS_INCS} 
    INTERFACE
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(splat 
    PUBLIC 
        ${ALL_DEPS_LIBS}
)

if(WIN32)
    target_compile_definitions(splat PUBLIC _USE_MATH_DEFINES NOMINMAX)
endif()

file(GLOB_RECURSE _headers ${PROJECT_SOURCE_DIR}/include/*.h)
target_sources(splat PRIVATE ${_headers})

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(VERSION_DIR "${CMAKE_BINARY_DIR}/splat")

install(TARGETS splat
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
