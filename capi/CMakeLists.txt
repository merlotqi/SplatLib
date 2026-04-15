include(GNUInstallDirs)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/splat_c.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/splat_c.h
    @ONLY
)

if(BUILD_SPLAT_SHARED)
  add_library(splat_c SHARED
      splat_c.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/splat_c.h
  )
else()
  add_library(splat_c STATIC
      splat_c.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/splat_c.h
  )
endif()

if(BUILD_SPLAT_SHARED AND WIN32)
    set_target_properties(splat_c PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

add_library(SPLAT::splat_c ALIAS splat_c)

target_include_directories(splat_c
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(splat_c
    PUBLIC
        splat
)

install(TARGETS splat_c
    EXPORT SplatTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/splat_c.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
