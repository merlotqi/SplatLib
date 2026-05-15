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
target_link_libraries(playcanvas_viewer_camera PUBLIC playcanvas_viewer_adapter Eigen3::Eigen)

# OpenGL renderer target
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
add_library(playcanvas_viewer_renderer STATIC gsplat_renderer.cpp)
target_include_directories(playcanvas_viewer_renderer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(playcanvas_viewer_renderer PUBLIC playcanvas_viewer_adapter playcanvas_viewer_camera GLEW::GLEW OpenGL::GL)

# PlaycanvasViewer executable
find_package(glfw3 CONFIG REQUIRED)
add_executable(PlaycanvasViewer main.cpp)
target_link_libraries(PlaycanvasViewer PRIVATE playcanvas_viewer_adapter playcanvas_viewer_camera playcanvas_viewer_renderer glfw GLEW::GLEW OpenGL::GL)
message(STATUS "PlaycanvasViewer executable configured.")
