add_executable(SplatKdTreeTests kdtree_test.cpp)
target_link_libraries(SplatKdTreeTests PRIVATE SPLAT::splat)

add_test(NAME SplatKdTreeTests COMMAND SplatKdTreeTests)
