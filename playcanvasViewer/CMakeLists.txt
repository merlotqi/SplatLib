cmake_minimum_required(VERSION 3.14)

# Adapter target first
add_library(playcanvas_viewer_adapter STATIC
    gsplat_data_adapter.cpp
)
target_include_directories(playcanvas_viewer_adapter
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(playcanvas_viewer_adapter
    PUBLIC
        SPLAT::splat
        Eigen3::Eigen
)

# Camera controller target after adapter exists
add_library(playcanvas_viewer_camera STATIC
    camera_controller.cpp
)
target_include_directories(playcanvas_viewer_camera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(playcanvas_viewer_camera PUBLIC SPLAT::splat Eigen3::Eigen)

find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

# PlaycanvasViewer executable
find_package(glfw3 CONFIG REQUIRED)
add_executable(PlaycanvasViewer main.cpp)
target_link_libraries(PlaycanvasViewer PRIVATE playcanvas_viewer_camera SPLAT::splat glfw GLEW::GLEW OpenGL::GL)
message(STATUS "PlaycanvasViewer executable configured.")
