#[=======================================================================[.rst:
FindWebP
--------

Find the WebP image library.

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` target:

``WebP::webp``
  The WebP library, if found.

Result variables
^^^^^^^^^^^^^^^^

``WebP_FOUND``
  True if the WebP library was found.
``WebP_INCLUDE_DIRS``
  Include directories for WebP.
``WebP_LIBRARIES``
  Libraries to link against.

#]=======================================================================]

find_path(WebP_INCLUDE_DIR
  NAMES webp/decode.h
  PATH_SUFFIXES include
)

find_library(WebP_LIBRARY
  NAMES webp libwebp
)

find_library(WebP_SHARPYUV_LIBRARY
  NAMES sharpyuv libsharpyuv
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(WebP
  REQUIRED_VARS WebP_LIBRARY WebP_INCLUDE_DIR
)

if(WebP_FOUND AND NOT TARGET WebP::webp)
  add_library(WebP::webp UNKNOWN IMPORTED)
  set_target_properties(WebP::webp PROPERTIES
    IMPORTED_LOCATION "${WebP_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${WebP_INCLUDE_DIR}"
  )
  if(WebP_SHARPYUV_LIBRARY)
    set_property(TARGET WebP::webp APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES "${WebP_SHARPYUV_LIBRARY}")
  endif()
endif()

mark_as_advanced(WebP_INCLUDE_DIR WebP_LIBRARY WebP_SHARPYUV_LIBRARY)
