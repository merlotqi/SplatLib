find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

add_executable(SplatViewer 
  application.cpp
  application.h
  camera.cpp
  camera.h
  main.cpp
  shadermanager.cpp
  shadermanager.h
  splatrenderer.cpp
  splatrenderer.h
)

target_link_libraries(SplatViewer 
    SPLAT::splat 
    ${GLEW_LIBRARIES}
    glfw
    ${OPENGL_LIBRARIES}
    glm::glm
    imgui
)

target_include_directories(SplatViewer
  PRIVATE
  ${IMGUI_DIR}
  ${IMGUI_DIR}/backends
)

target_compile_definitions(SplatViewer PRIVATE 
    GLEW_STATIC
    IMGUI_IMPL_OPENGL_LOADER_GLEW
)
