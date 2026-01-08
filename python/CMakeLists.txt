pybind11_add_module(splat_module binding.cpp)

target_link_libraries(splat_module PRIVATE SPLAT::splat)

set_target_properties(splat_module PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_lib"
)
