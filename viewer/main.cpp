/**
 * @file main.cpp
 * @brief Splat Viewer - OpenGL + ImGui viewer for 3D Gaussian Splatting data.
 *
 * A lightweight viewer for visualizing 3D Gaussian Splatting scenes.
 * Supports PLY, .splat, SOG, SPZ, KSplat, and LCC formats.
 *
 * Controls:
 * - Left mouse drag: Rotate camera
 * - Right mouse drag: Pan camera
 * - Scroll wheel: Zoom in/out
 * - O: Open file dialog
 * - R: Reset camera
 * - 1: Toggle stats panel
 * - 2: Toggle controls panel
 */

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <splat/splat.h>

#include <cctype>
#include <filesystem>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <string>

#include "camera.h"
#include "renderer.h"
#include "viewer_ui.h"


#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;
using namespace viewer;

// ============================================================================
// Global state
// ============================================================================
static glm::mat4 projectionMatrix{1.0f};
static int windowWidth = 1280;
static int windowHeight = 720;
static bool mouseLeftDown = false;
static bool mouseRightDown = false;
static float lastMouseX = 0.0f;
static float lastMouseY = 0.0f;
static bool wantQuit = false;
static char statusMessage[256] = "";
static float statusMessageTimer = 0.0f;

// ============================================================================
// File dialog (simple platform-specific implementation)
// ============================================================================
std::filesystem::path openFileDialog() {
#ifdef _WIN32
  OPENFILENAME ofn;
  char szFile[260] = {0};
  ZeroMemory(&ofn, sizeof(ofn));
  ofn.lStructSize = sizeof(ofn);
  ofn.hwndOwner = nullptr;
  ofn.lpstrFile = szFile;
  ofn.nMaxFile = sizeof(szFile);
  ofn.lpstrFilter = "Splat Files\0*.ply;*.splat;*.sog;*.spz;*.ksplat\0All Files\0*.*\0";
  ofn.nFilterIndex = 1;
  ofn.lpstrFileTitle = nullptr;
  ofn.nMaxFileTitle = 0;
  ofn.lpstrInitialDir = nullptr;
  ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

  if (GetOpenFileName(&ofn) == TRUE) {
    return std::filesystem::path(szFile);
  }
  return {};
#else
  std::string path;
  std::cout << "\nEnter file path: ";
  std::getline(std::cin, path);
  return std::filesystem::path(path);
#endif
}

// ============================================================================
// Load splat file
// ============================================================================
bool loadFile(Renderer& renderer, ViewerUIState& state, Camera& camera, const std::filesystem::path& filename) {
  try {
    std::unique_ptr<splat::DataTable> dataTable;
    std::string ext = filename.extension().u8string();
    std::cout << "Loading file: " << filename.u8string() << std::endl;
    std::cout << "Extension: " << ext << std::endl;

    for (char& c : ext) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    if (ext == ".ply") {
      std::cout << "Reading PLY file..." << std::endl;
      dataTable = splat::readPly(filename);
    } else if (ext == ".splat") {
      std::cout << "Reading .splat file..." << std::endl;
      dataTable = splat::readSplat(filename);
    } else if (ext == ".sog") {
      std::cout << "Reading SOG file..." << std::endl;
      dataTable = splat::readSog(filename, filename);
    } else if (ext == ".spz") {
      std::cout << "Reading SPZ file..." << std::endl;
      dataTable = splat::readSpz(filename);
    } else if (ext == ".ksplat") {
      std::cout << "Reading KSplat file..." << std::endl;
      dataTable = splat::readKsplat(filename);
    } else {
      snprintf(statusMessage, sizeof(statusMessage), "Unsupported file format: %s", ext.c_str());
      statusMessageTimer = 3.0f;
      std::cerr << "Unsupported format: " << ext << std::endl;
      return false;
    }

    if (!dataTable) {
      snprintf(statusMessage, sizeof(statusMessage), "Failed to load data");
      statusMessageTimer = 3.0f;
      std::cerr << "Failed to load data" << std::endl;
      return false;
    }

    std::cout << "Data loaded: " << dataTable->getNumRows() << " rows, " << dataTable->getNumColumns() << " columns"
              << std::endl;
    std::cout << "Column names: ";
    for (size_t i = 0; i < dataTable->getNumColumns(); ++i) {
      std::cout << dataTable->getColumn(i).name << " ";
    }
    std::cout << std::endl;

    if (dataTable->getNumRows() == 0) {
      snprintf(statusMessage, sizeof(statusMessage), "No splat data found");
      statusMessageTimer = 3.0f;
      return false;
    }

    // Compute scene bounds for camera setup
    const auto& xCol = dataTable->getColumnByName("x");
    const auto& yCol = dataTable->getColumnByName("y");
    const auto& zCol = dataTable->getColumnByName("z");

    float minX = xCol.getValue<float>(0), maxX = minX;
    float minY = yCol.getValue<float>(0), maxY = minY;
    float minZ = zCol.getValue<float>(0), maxZ = minZ;

    for (size_t i = 1; i < dataTable->getNumRows(); ++i) {
      minX = std::min(minX, xCol.getValue<float>(i));
      maxX = std::max(maxX, xCol.getValue<float>(i));
      minY = std::min(minY, yCol.getValue<float>(i));
      maxY = std::max(maxY, yCol.getValue<float>(i));
      minZ = std::min(minZ, zCol.getValue<float>(i));
      maxZ = std::max(maxZ, zCol.getValue<float>(i));
    }

    std::cout << "Bounds: X[" << minX << ", " << maxX << "], Y[" << minY << ", " << maxY << "], Z[" << minZ << ", "
              << maxZ << "]" << std::endl;

    glm::vec3 center((minX + maxX) * 0.5f, (minY + maxY) * 0.5f, (minZ + maxZ) * 0.5f);
    float extent = std::max({maxX - minX, maxY - minY, maxZ - minZ, 1.0f});

    renderer.loadSplats(*dataTable);
    state.currentFile = filename;

    // Setup camera
    camera.setTarget(center);
    camera.radius = glm::clamp(extent * 3.0f, 1.0f, 10000.0f);
    camera.theta = 0.0f;
    camera.phi = glm::pi<float>() / 3.0f;  // 60 degrees from above

    snprintf(statusMessage, sizeof(statusMessage), "Loaded %zu splats", renderer.getSplatCount());
    statusMessageTimer = 3.0f;
    return true;
  } catch (const std::exception& e) {
    snprintf(statusMessage, sizeof(statusMessage), "Error: %s", e.what());
    statusMessageTimer = 5.0f;
    std::cerr << "Error: " << e.what() << std::endl;
    return false;
  }
}

// ============================================================================
// Input callbacks
// ============================================================================
void framebufferSizeCallback(GLFWwindow*, int width, int height) {
  windowWidth = width;
  windowHeight = height;
  glViewport(0, 0, width, height);
}

void mouseButtonCallback(GLFWwindow*, int button, int action, int) {
  if (ImGui::GetIO().WantCaptureMouse) return;

  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    mouseLeftDown = (action == GLFW_PRESS);
  } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    mouseRightDown = (action == GLFW_PRESS);
  }
}

void cursorPosCallback(GLFWwindow*, double xpos, double ypos) {
  if (ImGui::GetIO().WantCaptureMouse) return;

  float dx = static_cast<float>(xpos) - lastMouseX;
  float dy = static_cast<float>(ypos) - lastMouseY;
  lastMouseX = static_cast<float>(xpos);
  lastMouseY = static_cast<float>(ypos);

  if (mouseLeftDown) {
    // Don't call mutable camera methods here - handle in main loop
  }
}

void scrollCallback(GLFWwindow* window, double, double yoffset) {
  if (ImGui::GetIO().WantCaptureMouse) return;
  Camera& cam = *static_cast<Camera*>(glfwGetWindowUserPointer(window));
  cam.zoom(-static_cast<float>(yoffset));
}

// ============================================================================
// Main entry point
// ============================================================================
int main(int argc, char** argv) {
  // Parse command line for file argument
  std::filesystem::path initialFile;
  for (int i = 1; i < argc; ++i) {
    initialFile = std::filesystem::path(argv[i]);
  }

  // Initialize GLFW
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  // Create window
  GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "Splat Viewer", nullptr, nullptr);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);
  glfwSetCursorPosCallback(window, cursorPosCallback);
  glfwSetScrollCallback(window, scrollCallback);
  glfwSwapInterval(1);  // VSync

  // Initialize GLEW
  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW" << std::endl;
    return -1;
  }

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();

  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 410");

  // Initialize viewer components
  Renderer renderer;
  renderer.initialize();

  Camera camera;
  ViewerUIState state;

  // Set camera user pointer for scroll callback
  glfwSetWindowUserPointer(window, &camera);

  // Load initial file if provided
  if (!initialFile.empty()) {
    loadFile(renderer, state, camera, initialFile);
  }

  // Main loop
  while (!glfwWindowShouldClose(window) && !wantQuit) {
    glfwPollEvents();

    // Keyboard input
    if (io.WantCaptureKeyboard) {
      // Reset last mouse pos to avoid jumps
      double lx, ly;
      glfwGetCursorPos(window, &lx, &ly);
      lastMouseX = static_cast<float>(lx);
      lastMouseY = static_cast<float>(ly);
    } else {
      int keyState[GLFW_KEY_LAST + 1] = {0};
      for (int key = 0; key <= GLFW_KEY_LAST; ++key) {
        keyState[key] = glfwGetKey(window, key);
      }

      static bool prevO = false;
      bool oPressed = (keyState[GLFW_KEY_O] == GLFW_PRESS);
      if (oPressed && !prevO) {
        std::filesystem::path file = openFileDialog();
        if (!file.empty()) {
          loadFile(renderer, state, camera, file);
        }
      }
      prevO = oPressed;

      if (keyState[GLFW_KEY_R] == GLFW_PRESS) {
        camera.theta = 0.0f;
        camera.phi = 1.0f;
        camera.radius = 5.0f;
        snprintf(statusMessage, sizeof(statusMessage), "Camera reset");
        statusMessageTimer = 2.0f;
      }

      if (keyState[GLFW_KEY_1] == GLFW_PRESS) {
        state.showStats = !state.showStats;
      }

      if (keyState[GLFW_KEY_2] == GLFW_PRESS) {
        state.showControls = !state.showControls;
      }

      // Mouse rotation
      double mx, my;
      glfwGetCursorPos(window, &mx, &my);
      float dx = static_cast<float>(mx) - lastMouseX;
      float dy = static_cast<float>(my) - lastMouseY;
      lastMouseX = static_cast<float>(mx);
      lastMouseY = static_cast<float>(my);

      if (mouseLeftDown) {
        camera.rotate(dx * 0.01f, dy * 0.01f);
      } else if (mouseRightDown) {
        camera.pan(dx, -dy);
      }
    }

    // Get framebuffer size for HiDPI
    int fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    float scale = static_cast<float>(fbWidth) / windowWidth;

    // Render
    glClearColor(state.bgR, state.bgG, state.bgB, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Setup projection
    float aspect = static_cast<float>(windowWidth) / windowHeight;
    projectionMatrix = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 1000.0f);

    // Render scene
    if (renderer.hasSplats()) {
      glm::mat4 vp = projectionMatrix * camera.getViewMatrix();
      renderer.render(vp, camera.getPosition());
    }

    // Build ImGui UI
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    renderUIOverlay(state, camera, renderer, renderer.hasSplats());

    // Status message toast
    if (statusMessageTimer > 0.0f) {
      statusMessageTimer -= ImGui::GetIO().DeltaTime;
      ImVec2 center(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y - 50.0f);
      ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
      ImGui::Begin("Status", nullptr,
                   ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize |
                       ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground);
      ImGui::TextColored(ImVec4(1, 1, 1, 1), "%s", statusMessage);
      ImGui::End();
    }

    if (!renderer.hasSplats()) {
      renderOpenPrompt();
    }

    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwMakeContextCurrent(window);
    glfwSwapBuffers(window);
  }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
