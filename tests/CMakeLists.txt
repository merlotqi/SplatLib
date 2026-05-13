add_executable(SplatKdTreeTests kdtree_test.cpp)
target_link_libraries(SplatKdTreeTests PRIVATE SPLAT::splat)

add_executable(SplatDecimateTests decimate_test.cpp)
target_link_libraries(SplatDecimateTests PRIVATE SPLAT::splat)

add_test(NAME SplatKdTreeTests COMMAND SplatKdTreeTests)
add_test(NAME SplatDecimateTests COMMAND SplatDecimateTests)
