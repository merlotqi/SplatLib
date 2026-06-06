// PlaycanvasViewer main.cpp
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <splat/splat.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "camera_controller.h"
#include "gsplat_data.h"
#include "gsplat_gl_renderer.h"
#include "splat/models/splatcloud.h"

// Helper: print usage
void print_usage() {
  std::cout << "Usage: PlaycanvasViewer [--width=N] [--height=N] [--max-splats=N] [--no-sort] <input-file>\n"
            << "Supported extensions: .ply, .splat, .sog, .spz, .ksplat\n"
            << "  --width=N        Set window width (default 1280)\n"
            << "  --height=N       Set window height (default 720)\n"
            << "  --max-splats=N   Limit number of splats rendered\n"
            << "  --no-sort        Disable back-to-front sorting, faster but visibly lower quality\n"
            << "  -h, --help       Show this help message\n";
}

struct ViewerOptions {
  int width = 1280;
  int height = 720;
  size_t maxSplats = 0;  // 0 means unlimited
  bool sort = true;
  std::filesystem::path inputFile;
};

size_t parseSizeOption(const std::string& optionName, const std::string& value) {
  if (value.empty() || value.front() == '-') {
    throw std::runtime_error("Invalid value for " + optionName + ": '" + value + "'");
  }

  size_t parsedChars = 0;
  size_t parsedValue = 0;
  try {
    parsedValue = static_cast<size_t>(std::stoull(value, &parsedChars));
  } catch (const std::exception&) {
    throw std::runtime_error("Invalid value for " + optionName + ": '" + value + "'");
  }

  if (parsedChars != value.size()) {
    throw std::runtime_error("Invalid value for " + optionName + ": '" + value + "'");
  }

  return parsedValue;
}

int parseDimensionOption(const std::string& optionName, const std::string& value) {
  const size_t parsedValue = parseSizeOption(optionName, value);
  if (parsedValue == 0 || parsedValue > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("Invalid value for " + optionName + ": '" + value + "'");
  }
  return static_cast<int>(parsedValue);
}

ViewerOptions parse_args(int argc, char** argv) {
  ViewerOptions opts;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage();
      std::exit(0);
    } else if (arg.rfind("--width=", 0) == 0) {
      opts.width = parseDimensionOption("--width", arg.substr(8));
    } else if (arg.rfind("--height=", 0) == 0) {
      opts.height = parseDimensionOption("--height", arg.substr(9));
    } else if (arg.rfind("--max-splats=", 0) == 0) {
      opts.maxSplats = parseSizeOption("--max-splats", arg.substr(14));
    } else if (arg == "--no-sort") {
      opts.sort = false;
    } else if (arg[0] == '-') {
      throw std::runtime_error("Unknown option: " + arg);
    } else {
      if (!opts.inputFile.empty()) throw std::runtime_error("Multiple input files specified");
      opts.inputFile = std::filesystem::path(arg);
    }
  }
  if (opts.inputFile.empty()) throw std::runtime_error("Missing input file");
  return opts;
}

// Data loading dispatch
std::unique_ptr<splat::SplatCloud> readInput(const std::filesystem::path& path) {
  if (!path.has_extension()) throw std::runtime_error("Input file has no extension");
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
  if (ext == ".ply") return splat::readPly(path.string());
  if (ext == ".splat") return splat::readSplat(path.string());
  if (ext == ".sog") return splat::readSog(path.string(), path.string());
  if (ext == ".spz") return splat::readSpz(path.string());
  if (ext == ".ksplat") return splat::readKsplat(path.string());
  throw std::runtime_error("Unsupported file extension: " + ext);
}

struct AppState {
  playcanvas_viewer::CameraController camera;
  splat::visualization::GSplatGLRenderer renderer;
  splat::visualization::GSplatGLRenderOptions renderOptions;
  bool dragging = false;
  double lastX = 0, lastY = 0;
  playcanvas_viewer::SceneBounds bounds;
};

// GLFW callbacks
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
  auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (state) {
    state->camera.resize(width, height);
    glViewport(0, 0, width, height);
  }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
  (void)mods;
  auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (!state) return;
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    if (action == GLFW_PRESS) {
      state->dragging = true;
      double x, y;
      glfwGetCursorPos(window, &x, &y);
      state->lastX = x;
      state->lastY = y;
    } else if (action == GLFW_RELEASE) {
      state->dragging = false;
    }
  }
}

void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos) {
  auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (!state || !state->dragging) return;
  double dx = xpos - state->lastX;
  double dy = ypos - state->lastY;
  state->camera.orbit(dx, dy);
  state->lastX = xpos;
  state->lastY = ypos;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
  (void)xoffset;
  auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (state) {
    state->camera.dolly(yoffset);
  }
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  (void)scancode;
  (void)mods;
  auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (!state) return;
  if (action == GLFW_PRESS) {
    if (key == GLFW_KEY_ESCAPE) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    } else if (key == GLFW_KEY_R) {
      state->camera.resetToBounds(state->bounds);
    }
  }
}

playcanvas_viewer::CameraInputState get_camera_input(GLFWwindow* window) {
  playcanvas_viewer::CameraInputState input;
  input.forward = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
  input.backward = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
  input.left = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
  input.right = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;
  input.up = glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS;
  input.down = glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS;
  return input;
}

splat::visualization::GSplatGLFrameState makeFrameState(const playcanvas_viewer::CameraController& camera, int width,
                                                        int height) {
  splat::visualization::GSplatGLFrameState frame;
  frame.view = camera.viewMatrix();
  frame.projection = camera.projectionMatrix();
  frame.cameraPosition = camera.position();
  frame.cameraForward = camera.forward();
  frame.width = width;
  frame.height = height;
  frame.nearPlane = camera.nearPlane();
  frame.farPlane = camera.farPlane();
  return frame;
}

// RAII for GLFW
struct GlfwSession {
  GlfwSession() {
    if (!glfwInit()) throw std::runtime_error("Failed to initialize GLFW");
  }
  ~GlfwSession() { glfwTerminate(); }
};

int main(int argc, char** argv) {
  try {
    auto options = parse_args(argc, argv);
    auto table = readInput(options.inputFile);
    if (!table || table->getNumRows() == 0) throw std::runtime_error("Input data is empty or failed to load");
    auto data = splat::visualization::adaptDataTableToGSplatData(*table, options.maxSplats);
    if (data.empty()) throw std::runtime_error("Adapted data is empty");
    if (!options.sort) {
      std::cerr << "PlaycanvasViewer warning: --no-sort disables alpha-depth ordering and reduces 3DGS quality.\n";
    }

    GlfwSession glfwSession;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    std::unique_ptr<GLFWwindow, decltype(&glfwDestroyWindow)> window(
        glfwCreateWindow(options.width, options.height, "PlaycanvasViewer", nullptr, nullptr), glfwDestroyWindow);
    if (!window) throw std::runtime_error("Failed to create GLFW window");
    glfwMakeContextCurrent(window.get());
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) throw std::runtime_error("Failed to initialize GLEW");
    glfwSwapInterval(1);  // vsync

    AppState state;
    state.camera.resize(options.width, options.height);
    state.bounds = playcanvas_viewer::computeSceneBounds(data);
    state.camera.resetToBounds(state.bounds);
    state.renderOptions.sortBackToFront = options.sort;
    state.renderer.setData(data);
    glfwSetWindowUserPointer(window.get(), &state);
    glfwSetFramebufferSizeCallback(window.get(), framebuffer_size_callback);
    glfwSetMouseButtonCallback(window.get(), mouse_button_callback);
    glfwSetCursorPosCallback(window.get(), cursor_pos_callback);
    glfwSetScrollCallback(window.get(), scroll_callback);
    glfwSetKeyCallback(window.get(), key_callback);

    auto lastTime = std::chrono::high_resolution_clock::now();
    while (!glfwWindowShouldClose(window.get())) {
      auto now = std::chrono::high_resolution_clock::now();
      float delta = std::chrono::duration<float>(now - lastTime).count();
      lastTime = now;
      playcanvas_viewer::CameraInputState input = get_camera_input(window.get());
      state.camera.fly(input, delta);
      int width, height;
      glfwGetFramebufferSize(window.get(), &width, &height);
      glViewport(0, 0, width, height);
      glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      state.renderer.render(makeFrameState(state.camera, width, height), state.renderOptions);
      glfwSwapBuffers(window.get());
      glfwPollEvents();
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "PlaycanvasViewer error: " << ex.what() << std::endl;
    return 1;
  }
}
