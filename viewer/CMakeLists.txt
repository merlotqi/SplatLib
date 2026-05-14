cmake_minimum_required(VERSION 3.14)

# ============================================================================
# Find vcpkg packages
# ============================================================================
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

# ============================================================================
# Dear ImGui
# ============================================================================
include(FetchContent)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.90.9
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
endif()

set(IMGUI_SOURCE_DIR ${imgui_SOURCE_DIR})

# ============================================================================
# Create imgui library target
# ============================================================================
set(IMGUI_SOURCES
    ${IMGUI_SOURCE_DIR}/imgui.cpp
    ${IMGUI_SOURCE_DIR}/imgui_demo.cpp
    ${IMGUI_SOURCE_DIR}/imgui_draw.cpp
    ${IMGUI_SOURCE_DIR}/imgui_tables.cpp
    ${IMGUI_SOURCE_DIR}/imgui_widgets.cpp
    ${IMGUI_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

add_library(imgui_lib STATIC ${IMGUI_SOURCES})
target_include_directories(imgui_lib PUBLIC
    ${IMGUI_SOURCE_DIR}
    ${IMGUI_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw GLEW::GLEW OpenGL::GL)

# ============================================================================
# Splat Viewer executable
# ============================================================================
set(VIEWER_SOURCES
    main.cpp
    viewer_ui.cpp
)

set(VIEWER_HEADERS
    viewer_ui.h
)

add_executable(splat_viewer ${VIEWER_SOURCES} ${VIEWER_HEADERS})

target_include_directories(splat_viewer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(splat_viewer PRIVATE
    splat
    imgui_lib
    glfw
    GLEW::GLEW
    OpenGL::GL
)

message(STATUS "Splat Viewer: Building ImGui controls with VTK visualization renderer")
