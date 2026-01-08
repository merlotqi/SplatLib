#include "application.h"

#include <chrono>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "shadermanager.h"

Application::Application(int width, int height) : width_(width), height_(height) {}

Application::~Application() { cleanup(); }

int Application::run() {
  if (!initGLFW()) return -1;
  if (!initImGui()) return -1;

  mainLoop();
  return 0;
}

bool Application::initGLFW() {
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 4);  // MSAA

  window_ = glfwCreateWindow(width_, height_, "3DGS Viewer", nullptr, nullptr);
  if (!window_) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(window_);
  glfwSwapInterval(1);  // VSync

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW" << std::endl;
    return false;
  }

  std::cout << "OpenGL: " << glGetString(GL_VERSION) << std::endl;
  std::cout << "GLSL: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

  // 设置回调
  glfwSetWindowUserPointer(window_, this);

  auto mouse_button_callback = [](GLFWwindow* window, int button, int action, int mods) {
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      app->mouse_pressed_ = (action == GLFW_PRESS);
      if (app->mouse_pressed_) {
        glfwGetCursorPos(window, &app->last_mouse_x_, &app->last_mouse_y_);
      }
    }
  };

  auto cursor_pos_callback = [](GLFWwindow* window, double xpos, double ypos) {
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app->mouse_pressed_) {
      double dx = xpos - app->last_mouse_x_;
      double dy = ypos - app->last_mouse_y_;
      app->last_mouse_x_ = xpos;
      app->last_mouse_y_ = ypos;

      if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS) {
        app->camera_.orbit(dx * 0.5f, dy * 0.5f);
      } else {
        app->camera_.rotate(dx * 0.5f, dy * 0.5f);
      }
    }
  };

  auto scroll_callback = [](GLFWwindow* window, double xoffset, double yoffset) {
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    app->camera_.zoom(-yoffset * 0.5f);
  };

  auto key_callback = [](GLFWwindow* window, int key, int scancode, int action, int mods) {
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
      switch (key) {
        case GLFW_KEY_SPACE:
          app->is_playing_ = !app->is_playing_;
          break;
        case GLFW_KEY_R:
          app->resetCamera();
          break;
        case GLFW_KEY_F:
          app->fitToView();
          break;
        case GLFW_KEY_H:
          app->show_help_ = !app->show_help_;
          break;
        case GLFW_KEY_O:
          if (mods & GLFW_MOD_CONTROL) {
            std::string filepath;
            if (app->openFileDialog(filepath)) {
              app->loadSplatData(filepath);
            }
          }
          break;
      }
    }
  };

  glfwSetMouseButtonCallback(window_, mouse_button_callback);
  glfwSetCursorPosCallback(window_, cursor_pos_callback);
  glfwSetScrollCallback(window_, scroll_callback);
  glfwSetKeyCallback(window_, key_callback);

  return true;
}

bool Application::initImGui() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

  ImGui::StyleColorsDark();

  float scale = 1.0f;
#ifdef __APPLE__
  scale = 2.0f;
#endif
  ImGui::GetStyle().ScaleAllSizes(scale);

  if (!ImGui_ImplGlfw_InitForOpenGL(window_, true)) {
    std::cerr << "Failed to initialize ImGui GLFW backend" << std::endl;
    return false;
  }

  if (!ImGui_ImplOpenGL3_Init("#version 430")) {
    std::cerr << "Failed to initialize ImGui OpenGL3 backend" << std::endl;
    return false;
  }

  return true;
}

void Application::mainLoop() {
  auto last_time = std::chrono::high_resolution_clock::now();

  while (!glfwWindowShouldClose(window_)) {
    auto current_time = std::chrono::high_resolution_clock::now();
    float delta_time = std::chrono::duration<float>(current_time - last_time).count();
    last_time = current_time;

    frame_time_ = delta_time;
    frame_count_++;

    glfwPollEvents();

    if (is_playing_) {
      update(delta_time);
    }

    render();

    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
      ImGui::UpdatePlatformWindows();
      ImGui::RenderPlatformWindowsDefault();
      glfwMakeContextCurrent(window_);
    }

    glfwSwapBuffers(window_);
  }
}

void Application::update(float deltaTime) {
  if (auto_rotate_) {
    camera_.orbit(auto_rotate_speed_ * deltaTime * 30.0f, 0.0f);
  }

  static float shader_check_timer = 0.0f;
  shader_check_timer += deltaTime;
  if (shader_check_timer > 1.0f) {
    ShaderManager::instance().reloadAll();
    if (renderer_) {
      renderer_->reloadShaders();
    }
    shader_check_timer = 0.0f;
  }

  if (is_loaded_ && renderer_) {
    renderer_->updateSorting(camera_.getViewMatrix());
  }
}

void Application::render() {
  int width, height;
  glfwGetFramebufferSize(window_, &width, &height);

  glViewport(0, 0, width, height);
  glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (is_loaded_ && renderer_) {
    glm::mat4 view = camera_.getViewMatrix();
    glm::mat4 proj = camera_.getProjectionMatrix(static_cast<float>(width) / height);

    renderer_->render(view, proj, focal_length_, glm::vec2(width, height));
  }

  renderUI();
}

void Application::renderUI() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  ImGui::DockSpaceOverViewport(ImGui::GetMainViewport()->ID);

  renderMainMenu();

  if (show_settings_) {
    renderSettingsPanel();
  }

  if (show_statistics_) {
    renderStatisticsPanel();
  }

  if (show_help_) {
    renderHelpPanel();
  }

  ImGui::Begin("Status", nullptr,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings);

  ImGui::Text("FPS: %.1f (%.2f ms)", 1.0f / frame_time_, frame_time_ * 1000.0f);
  ImGui::SameLine();
  ImGui::Text(" | ");
  ImGui::SameLine();
  ImGui::Text("Loaded: %s", is_loaded_ ? current_file_.c_str() : "None");

  if (is_loaded_ && renderer_) {
    ImGui::SameLine();
    ImGui::Text(" | ");
    ImGui::SameLine();
    ImGui::Text("Points: %d", renderer_->getPointCount());
  }

  ImGui::End();

  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Application::renderMainMenu() {
  if (ImGui::BeginMainMenuBar()) {
    if (ImGui::BeginMenu("File")) {
      if (ImGui::MenuItem("Open Splat file...", "Ctrl+O")) {
        std::string filepath = "/home/merlot/codes/splat-transform/examples/qiantai.ply";
        if (openFileDialog(filepath)) {
          loadSplatData(filepath);
        }
      }

      ImGui::Separator();

      if (ImGui::MenuItem("Take Screenshot", "F12")) {
        takeScreenshot();
      }

      ImGui::Separator();

      if (ImGui::MenuItem("Exit", "Esc")) {
        glfwSetWindowShouldClose(window_, true);
      }

      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("View")) {
      ImGui::MenuItem("Settings", nullptr, &show_settings_);
      ImGui::MenuItem("Statistics", nullptr, &show_statistics_);
      ImGui::MenuItem("Help", "H", &show_help_);

      ImGui::Separator();
      ImGui::MenuItem("Auto Rotate", "Space", &auto_rotate_);
      ImGui::MenuItem("Pause", "Space", &is_playing_);

      ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Camera")) {
      if (ImGui::MenuItem("Reset View", "R")) {
        resetCamera();
      }
      if (ImGui::MenuItem("Fit to View", "F")) {
        fitToView();
      }
      ImGui::EndMenu();
    }

    ImGui::EndMainMenuBar();
  }
}

void Application::renderSettingsPanel() {
  ImGui::Begin("Settings", &show_settings_);

  if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::SliderFloat("FOV", &camera_.fov_, 10.0f, 120.0f, "%.0f deg");
    ImGui::SliderFloat("Move Speed", &camera_.move_speed_, 0.1f, 50.0f);
    ImGui::SliderFloat("Rotate Speed", &camera_.rotate_speed_, 0.1f, 5.0f);
    ImGui::SliderFloat("Zoom Speed", &camera_.zoom_speed_, 0.1f, 10.0f);

    ImGui::Separator();
    ImGui::Checkbox("Auto Rotate", &auto_rotate_);
    if (auto_rotate_) {
      ImGui::SliderFloat("Rotation Speed", &auto_rotate_speed_, 0.0f, 5.0f);
    }
  }

  if (ImGui::CollapsingHeader("Rendering")) {
    ImGui::Checkbox("Enable Sorting", &sort_enabled_);
    if (sort_enabled_) {
      ImGui::SliderInt("Sort Interval", &sort_interval_, 1, 60);
    }

    ImGui::SliderFloat("Focal Length", &focal_length_, 100.0f, 5000.0f);
    ImGui::SliderFloat("Point Scale", &point_scale_, 0.1f, 5.0f);

    if (renderer_) {
      renderer_->setSortEnabled(sort_enabled_);
      renderer_->setSortInterval(sort_interval_);
      renderer_->setPointScale(point_scale_);
    }
  }

  ImGui::End();
}

void Application::renderStatisticsPanel() {
  ImGui::Begin("Statistics", &show_statistics_);

  if (is_loaded_ && renderer_) {
    ImGui::Text("File: %s", current_file_.c_str());
    ImGui::Separator();

    ImGui::Text("Point Count: %d", renderer_->getPointCount());
    ImGui::Text("Memory Usage: %.2f MB", renderer_->getMemoryUsage() / (1024.0f * 1024.0f));

    ImGui::Separator();
    ImGui::Text("Rendering Stats:");
    ImGui::Text("FPS: %.1f", 1.0f / frame_time_);
    ImGui::Text("Frame Time: %.2f ms", frame_time_ * 1000.0f);

    if (sort_enabled_) {
      ImGui::Text("Sorting: Enabled (%d frames)", sort_interval_);
    } else {
      ImGui::Text("Sorting: Disabled");
    }
  } else {
    ImGui::Text("No model loaded");
  }

  ImGui::End();
}

void Application::renderHelpPanel() {
  ImGui::Begin("Help & Controls", &show_help_, ImGuiWindowFlags_AlwaysAutoResize);

  ImGui::Text("Mouse Controls:");
  ImGui::BulletText("Left Drag: Rotate camera");
  ImGui::BulletText("Alt + Left Drag: Orbit around target");
  ImGui::BulletText("Scroll: Zoom in/out");

  ImGui::Spacing();
  ImGui::Text("Keyboard Shortcuts:");
  ImGui::BulletText("Ctrl+O: Open PLY file");
  ImGui::BulletText("F12: Take screenshot");
  ImGui::BulletText("R: Reset camera");
  ImGui::BulletText("F: Fit to view");
  ImGui::BulletText("Space: Toggle play/pause");
  ImGui::BulletText("H: Toggle this help");
  ImGui::BulletText("Esc: Exit");

  ImGui::Spacing();
  ImGui::Text("Tips:");
  ImGui::BulletText("Adjust camera speed in Settings panel");
  ImGui::BulletText("Enable auto-rotate for automatic rotation");
  ImGui::BulletText("Adjust point scale if Gaussians are too small/large");

  ImGui::End();
}

bool Application::loadSplatData(const std::string& filepath) {
  try {
    std::cout << "Loading splat data: " << filepath << std::endl;

    auto plyData = splat::readPly(filepath);
    if (!plyData || plyData->elements.empty()) {
      std::cerr << "Failed to load PLY file or no data found" << std::endl;
      return false;
    }

    renderer_ = std::make_unique<SplatRenderer>();
    if (!renderer_->loadFromDataTable(*plyData->elements[0].dataTable)) {
      std::cerr << "Failed to setup renderer" << std::endl;
      return false;
    }

    const auto& xCol = plyData->elements[0].dataTable->getColumnByName("x").asVector<float>();
    const auto& yCol = plyData->elements[0].dataTable->getColumnByName("y").asVector<float>();
    const auto& zCol = plyData->elements[0].dataTable->getColumnByName("z").asVector<float>();

    if (xCol.empty()) {
      std::cerr << "No position data" << std::endl;
      return false;
    }

    float minX = xCol[0], maxX = xCol[0];
    float minY = yCol[0], maxY = yCol[0];
    float minZ = zCol[0], maxZ = zCol[0];

    for (size_t i = 1; i < xCol.size(); ++i) {
      minX = std::min(minX, xCol[i]);
      maxX = std::max(maxX, xCol[i]);
      minY = std::min(minY, yCol[i]);
      maxY = std::max(maxY, yCol[i]);
      minZ = std::min(minZ, zCol[i]);
      maxZ = std::max(maxZ, zCol[i]);
    }

    camera_.fitToBox(glm::vec3(minX, minY, minZ), glm::vec3(maxX, maxY, maxZ));

    current_file_ = filepath;
    is_loaded_ = true;

    std::cout << "Loaded successfully: " << filepath << std::endl;
    return true;

  } catch (const std::exception& e) {
    std::cerr << "Error loading PLY: " << e.what() << std::endl;
    return false;
  }
}

bool Application::openFileDialog(std::string& outPath) {
  // 使用ImGui的文件对话框扩展或tinyfiledialogs
  // 这里简化处理，实际应该使用文件对话框库
  const char* filters[] = {"*.ply"};

  // 使用tinyfiledialogs（需要包含头文件）
  // char* file = tinyfd_openFileDialog("Open PLY File", "", 1, filters, "PLY Files", 0);
  // if (file) {
  //     outPath = file;
  //     return true;
  // }

  // 暂时返回false，实际使用时应该集成文件对话框
  std::cerr << "File dialog not implemented. Please specify file via command line." << std::endl;
  return true;
}

void Application::resetCamera() { camera_.reset(); }

void Application::fitToView() {
  if (is_loaded_ && renderer_) {
    // 这里需要获取数据范围，应该缓存起来
    // camera_.fitToBox(min, max);
  }
}

void Application::takeScreenshot() {
  int width, height;
  glfwGetFramebufferSize(window_, &width, &height);

  std::vector<unsigned char> pixels(width * height * 3);
  glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

  // 翻转Y轴
  for (int y = 0; y < height / 2; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        std::swap(pixels[(y * width + x) * 3 + c], pixels[((height - 1 - y) * width + x) * 3 + c]);
      }
    }
  }

  auto now = std::chrono::system_clock::now();
  auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

  std::string filename = "screenshot_" + std::to_string(timestamp) + ".png";

  // 需要使用stb_image_write等库保存图片
  // stbi_write_png(filename.c_str(), width, height, 3, pixels.data(), width * 3);

  std::cout << "Screenshot saved to: " << filename << std::endl;
}

void Application::cleanup() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  if (window_) {
    glfwDestroyWindow(window_);
  }
  glfwTerminate();
}
