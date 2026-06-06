#[=======================================================================[.rst:
FindZstd
--------

Find the Zstandard compression library.

Imported targets
^^^^^^^^^^^^^^^^

This module defines the following :prop_tgt:`IMPORTED` targets:

``zstd::libzstd_shared``
  The shared zstd library, if found.

``zstd::libzstd_static``
  The static zstd library, if found.

``zstd::libzstd``
  Alias that prefers shared, falls back to static.

Result variables
^^^^^^^^^^^^^^^^

``Zstd_FOUND``
  True if the zstd library was found.
``Zstd_INCLUDE_DIRS``
  Include directories for zstd.
``Zstd_LIBRARIES``
  Libraries to link against.

#]=======================================================================]

find_path(Zstd_INCLUDE_DIR
  NAMES zstd.h
  PATH_SUFFIXES include
)

find_library(Zstd_SHARED_LIBRARY
  NAMES zstd libzstd
)

find_library(Zstd_STATIC_LIBRARY
  NAMES libzstd_static.a zstd_static
)

include(FindPackageHandleStandardArgs)
# Accept either shared or static library
if(Zstd_SHARED_LIBRARY)
  set(_Zstd_LIB "${Zstd_SHARED_LIBRARY}")
elseif(Zstd_STATIC_LIBRARY)
  set(_Zstd_LIB "${Zstd_STATIC_LIBRARY}")
endif()
find_package_handle_standard_args(Zstd
  REQUIRED_VARS _Zstd_LIB Zstd_INCLUDE_DIR
)

if(Zstd_FOUND)
  # Shared library target
  if(Zstd_SHARED_LIBRARY AND NOT TARGET zstd::libzstd_shared)
    add_library(zstd::libzstd_shared UNKNOWN IMPORTED)
    set_target_properties(zstd::libzstd_shared PROPERTIES
      IMPORTED_LOCATION "${Zstd_SHARED_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${Zstd_INCLUDE_DIR}"
    )
  endif()

  # Static library target
  if(Zstd_STATIC_LIBRARY AND NOT TARGET zstd::libzstd_static)
    add_library(zstd::libzstd_static UNKNOWN IMPORTED)
    set_target_properties(zstd::libzstd_static PROPERTIES
      IMPORTED_LOCATION "${Zstd_STATIC_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${Zstd_INCLUDE_DIR}"
    )
  endif()

  # Generic alias (prefer shared, fall back to static)
  if(NOT TARGET zstd::libzstd)
    if(Zstd_SHARED_LIBRARY)
      add_library(zstd::libzstd ALIAS zstd::libzstd_shared)
    elseif(Zstd_STATIC_LIBRARY)
      add_library(zstd::libzstd ALIAS zstd::libzstd_static)
    endif()
  endif()
endif()

mark_as_advanced(Zstd_INCLUDE_DIR Zstd_SHARED_LIBRARY Zstd_STATIC_LIBRARY)
