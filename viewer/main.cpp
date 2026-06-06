/**
 * @file main.cpp
 * @brief ImGui control shell for the VTK-based splat visualizer.
 */

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <splat/splat.h>
#include <splat/visualization/splat_visualizer.h>
#include <vtkCamera.h>
#include <vtkMath.h>
#include <vtkRenderer.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gsplat_data.h"
#include "viewer_ui.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

constexpr const char* kCloudId = "viewer-splat";
constexpr double kInteractionSortDelaySeconds = 0.75;

int controlWindowWidth = 360;
int controlWindowHeight = 420;
bool wantQuit = false;
char statusMessage[256] = "";
float statusMessageTimer = 0.0f;
std::array<double, 6> currentCameraBounds{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
bool hasCameraBounds = false;

struct FlyNavigationState {
  bool forward = false;
  bool backward = false;
  bool left = false;
  bool right = false;
  bool down = false;
  bool up = false;
};

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

std::string lowerExtension(const std::filesystem::path& filename) {
  std::string ext = filename.extension().u8string();
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return ext;
}

std::unique_ptr<splat::SplatCloud> readSplatFile(const std::filesystem::path& filename) {
  const std::string ext = lowerExtension(filename);
  std::unique_ptr<splat::SplatCloud> dataTable;

  if (ext == ".ply") {
    dataTable = splat::readPly(filename);
  } else if (ext == ".splat") {
    dataTable = splat::readSplat(filename);
  } else if (ext == ".sog") {
    dataTable = splat::readSog(filename, filename);
  } else if (ext == ".spz") {
    dataTable = splat::readSpz(filename);
  } else if (ext == ".ksplat") {
    dataTable = splat::readKsplat(filename);
  } else {
    throw std::runtime_error("Unsupported file format: " + ext);
  }

  return dataTable;
}

splat::SplatRenderOptions makeRenderOptions(const viewer::ViewerUIState& state) {
  splat::SplatRenderOptions options;
  options.globalOpacity = state.opacity;
  options.sizeScale = state.sizeScale;
  options.maxPointSize = state.maxPointSize;
  options.sortBackToFront = state.sortBackToFront;
  options.clampColors = state.clampColors;
  return options;
}

bool sameRenderOptions(const splat::SplatRenderOptions& lhs, const splat::SplatRenderOptions& rhs) {
  return lhs.globalOpacity == rhs.globalOpacity && lhs.sizeScale == rhs.sizeScale &&
         lhs.maxPointSize == rhs.maxPointSize && lhs.sortBackToFront == rhs.sortBackToFront &&
         lhs.clampColors == rhs.clampColors && lhs.freezeSortOrder == rhs.freezeSortOrder;
}

bool isKeyPressed(GLFWwindow* window, int key) { return glfwGetKey(window, key) == GLFW_PRESS; }

double boundsRadius(const std::array<double, 6>& bounds) {
  const double dx = bounds[1] - bounds[0];
  const double dy = bounds[3] - bounds[2];
  const double dz = bounds[5] - bounds[4];
  return std::max(1e-3, 0.5 * std::sqrt(dx * dx + dy * dy + dz * dz));
}

std::array<double, 3> boundsCenter(const std::array<double, 6>& bounds) {
  return {
      0.5 * (bounds[0] + bounds[1]),
      0.5 * (bounds[2] + bounds[3]),
      0.5 * (bounds[4] + bounds[5]),
  };
}

void setInteriorClippingRange(splat::SplatVisualizer& visualizer, const std::array<double, 6>& bounds) {
  auto* renderer = visualizer.getRenderer();
  if (renderer == nullptr) {
    return;
  }

  auto* camera = renderer->GetActiveCamera();
  const double radius = boundsRadius(bounds);
  const double nearClip = std::max(radius * 0.0002, 0.002);
  const double farClip = std::max(radius * 64.0, nearClip * 1000.0);
  camera->SetClippingRange(nearClip, farClip);
}

void placeCameraInsideBounds(splat::SplatVisualizer& visualizer, const std::array<double, 6>& bounds) {
  auto* renderer = visualizer.getRenderer();
  if (renderer == nullptr) {
    return;
  }

  auto* camera = renderer->GetActiveCamera();
  double position[3];
  double focalPoint[3];
  camera->GetPosition(position);
  camera->GetFocalPoint(focalPoint);

  double forward[3] = {focalPoint[0] - position[0], focalPoint[1] - position[1], focalPoint[2] - position[2]};
  if (vtkMath::Normalize(forward) == 0.0) {
    forward[0] = 0.0;
    forward[1] = 0.0;
    forward[2] = -1.0;
  }

  const auto center = boundsCenter(bounds);
  const double radius = boundsRadius(bounds);
  camera->SetPosition(center[0], center[1], center[2]);
  camera->SetFocalPoint(center[0] + forward[0] * radius * 0.25, center[1] + forward[1] * radius * 0.25,
                        center[2] + forward[2] * radius * 0.25);
  camera->OrthogonalizeViewUp();
  setInteriorClippingRange(visualizer, bounds);
}

bool applyFlyNavigation(splat::SplatVisualizer& visualizer, const viewer::ViewerUIState& state,
                        const FlyNavigationState& keyState, GLFWwindow* controlWindow, double deltaSeconds) {
  if (!visualizer.contains(kCloudId)) {
    return false;
  }

  FlyNavigationState combined = keyState;
  combined.forward = combined.forward || isKeyPressed(controlWindow, GLFW_KEY_W);
  combined.backward = combined.backward || isKeyPressed(controlWindow, GLFW_KEY_S);
  combined.left = combined.left || isKeyPressed(controlWindow, GLFW_KEY_A);
  combined.right = combined.right || isKeyPressed(controlWindow, GLFW_KEY_D);
  combined.down = combined.down || isKeyPressed(controlWindow, GLFW_KEY_Q);
  combined.up = combined.up || isKeyPressed(controlWindow, GLFW_KEY_E);

  const double forwardAmount = static_cast<double>(combined.forward) - static_cast<double>(combined.backward);
  const double rightAmount = static_cast<double>(combined.right) - static_cast<double>(combined.left);
  const double upAmount = static_cast<double>(combined.up) - static_cast<double>(combined.down);
  if (forwardAmount == 0.0 && rightAmount == 0.0 && upAmount == 0.0) {
    return false;
  }

  auto* renderer = visualizer.getRenderer();
  if (renderer == nullptr) {
    return false;
  }

  auto* camera = renderer->GetActiveCamera();
  double position[3];
  double focalPoint[3];
  double viewUp[3];
  camera->GetPosition(position);
  camera->GetFocalPoint(focalPoint);
  camera->GetViewUp(viewUp);

  double forward[3] = {focalPoint[0] - position[0], focalPoint[1] - position[1], focalPoint[2] - position[2]};
  if (vtkMath::Normalize(forward) == 0.0) {
    return false;
  }
  vtkMath::Normalize(viewUp);
  double right[3];
  vtkMath::Cross(forward, viewUp, right);
  if (vtkMath::Normalize(right) == 0.0) {
    return false;
  }

  double move[3] = {
      forward[0] * forwardAmount + right[0] * rightAmount + viewUp[0] * upAmount,
      forward[1] * forwardAmount + right[1] * rightAmount + viewUp[1] * upAmount,
      forward[2] * forwardAmount + right[2] * rightAmount + viewUp[2] * upAmount,
  };
  vtkMath::Normalize(move);
  const double distance = std::max(0.0, static_cast<double>(state.flySpeed)) * std::max(deltaSeconds, 0.001);
  for (double& value : move) {
    value *= distance;
  }

  camera->SetPosition(position[0] + move[0], position[1] + move[1], position[2] + move[2]);
  camera->SetFocalPoint(focalPoint[0] + move[0], focalPoint[1] + move[1], focalPoint[2] + move[2]);
  if (hasCameraBounds) {
    setInteriorClippingRange(visualizer, currentCameraBounds);
  }
  return true;
}

std::pair<float, float> robustRange(std::vector<float>& values) {
  std::sort(values.begin(), values.end());
  if (values.empty()) {
    return {0.0f, 0.0f};
  }

  size_t trim = 0;
  if (values.size() >= 1000) {
    trim = std::max<size_t>(1, values.size() / 1000);
  }

  const size_t low = std::min(trim, values.size() - 1);
  const size_t high = std::max(low, values.size() - 1 - trim);
  return {values[low], values[high]};
}

float robustScalePadding(std::vector<float>& values) {
  std::sort(values.begin(), values.end());
  if (values.empty()) {
    return 0.0f;
  }

  size_t index = values.size() - 1;
  if (values.size() >= 1000) {
    index = std::min(values.size() - 1, values.size() - 1 - values.size() / 1000);
  }

  return values[index] * 2.0f;
}

std::array<double, 6> computeCameraBounds(const splat::SplatCloud& dataTable) {
  const auto renderData = splat::visualization::adaptDataTableToGSplatData(dataTable);
  if (renderData.empty()) {
    return {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
  }

  std::vector<float> xs;
  std::vector<float> ys;
  std::vector<float> zs;
  std::vector<float> maxScales;
  xs.reserve(renderData.size());
  ys.reserve(renderData.size());
  zs.reserve(renderData.size());
  maxScales.reserve(renderData.size());

  for (std::size_t i = 0; i < renderData.size(); ++i) {
    const auto& center = renderData.centers[i];
    const auto& scale = renderData.scales[i];
    const float maxScale = std::max({scale.x, scale.y, scale.z});
    if (std::isfinite(center.x) && std::isfinite(center.y) && std::isfinite(center.z) && std::isfinite(maxScale)) {
      xs.push_back(center.x);
      ys.push_back(center.y);
      zs.push_back(center.z);
      maxScales.push_back(maxScale);
    }
  }

  if (xs.empty()) {
    return {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
  }

  const auto [minX, maxX] = robustRange(xs);
  const auto [minY, maxY] = robustRange(ys);
  const auto [minZ, maxZ] = robustRange(zs);
  const float padding = robustScalePadding(maxScales);

  return {static_cast<double>(minX - padding), static_cast<double>(maxX + padding),
          static_cast<double>(minY - padding), static_cast<double>(maxY + padding),
          static_cast<double>(minZ - padding), static_cast<double>(maxZ + padding)};
}

void resetCameraToBounds(splat::SplatVisualizer& visualizer, const std::array<double, 6>& bounds) {
  visualizer.resetCameraToBounds(bounds.data());
}

bool loadFile(splat::SplatVisualizer& visualizer, viewer::ViewerUIState& state, const std::filesystem::path& filename) {
  try {
    std::cout << "Loading file: " << filename.u8string() << std::endl;
    auto dataTable = readSplatFile(filename);
    if (!dataTable || dataTable->getNumRows() == 0) {
      std::snprintf(statusMessage, sizeof(statusMessage), "No splat data found");
      statusMessageTimer = 3.0f;
      return false;
    }

    currentCameraBounds = computeCameraBounds(*dataTable);
    hasCameraBounds = true;
    state.flySpeed = static_cast<float>(std::clamp(boundsRadius(currentCameraBounds) * 0.25, 0.05, 20.0));

    auto sharedData = std::shared_ptr<const splat::SplatCloud>(std::move(dataTable));
    const auto options = makeRenderOptions(state);
    const bool ok = visualizer.contains(kCloudId) ? visualizer.updateSplatCloud(sharedData, kCloudId, options)
                                                  : visualizer.addSplatCloud(sharedData, kCloudId, options);
    if (!ok) {
      std::snprintf(statusMessage, sizeof(statusMessage), "Failed to upload splats");
      statusMessageTimer = 3.0f;
      return false;
    }

    state.currentFile = filename;
    state.splatCount = visualizer.getSplatCount(kCloudId);
    resetCameraToBounds(visualizer, currentCameraBounds);

    std::cout << "Loaded " << state.splatCount << " splats" << std::endl;
    std::snprintf(statusMessage, sizeof(statusMessage), "Loaded %zu splats", state.splatCount);
    statusMessageTimer = 3.0f;
    return true;
  } catch (const std::exception& e) {
    std::snprintf(statusMessage, sizeof(statusMessage), "Error: %s", e.what());
    statusMessageTimer = 5.0f;
    std::cerr << "Error: " << e.what() << std::endl;
    return false;
  }
}

void framebufferSizeCallback(GLFWwindow*, int width, int height) {
  controlWindowWidth = std::max(width, 1);
  controlWindowHeight = std::max(height, 1);
  glViewport(0, 0, controlWindowWidth, controlWindowHeight);
}

void renderStatusToast() {
  if (statusMessageTimer <= 0.0f) {
    return;
  }

  statusMessageTimer -= ImGui::GetIO().DeltaTime;
  ImVec2 center(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y - 42.0f);
  ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
  ImGui::Begin("Status", nullptr,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize |
                   ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBackground);
  ImGui::TextColored(ImVec4(1, 1, 1, 1), "%s", statusMessage);
  ImGui::End();
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path initialFile;
  for (int i = 1; i < argc; ++i) {
    initialFile = std::filesystem::path(argv[i]);
  }

  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  GLFWwindow* controlWindow =
      glfwCreateWindow(controlWindowWidth, controlWindowHeight, "Splat Viewer Controls", nullptr, nullptr);
  if (!controlWindow) {
    std::cerr << "Failed to create GLFW control window" << std::endl;
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(controlWindow);
  glfwSetFramebufferSizeCallback(controlWindow, framebufferSizeCallback);
  glfwSwapInterval(1);

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW" << std::endl;
    glfwDestroyWindow(controlWindow);
    glfwTerminate();
    return -1;
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  ImGui_ImplGlfw_InitForOpenGL(controlWindow, true);
  ImGui_ImplOpenGL3_Init("#version 410");

  viewer::ViewerUIState state;
  FlyNavigationState flyKeys;
  splat::SplatVisualizer visualizer("Splat Viewer");
  visualizer.setWindowSize(1280, 720);
  visualizer.setBackgroundColor(state.bgR, state.bgG, state.bgB);
  visualizer.setAxesEnabled(state.showAxes);
  visualizer.setDefaultHotkeysEnabled(false);
  visualizer.setDefaultMouseWheelEnabled(false);
  visualizer.registerKeyCallback([&flyKeys, &visualizer](const splat::KeyEvent& event) {
    const bool pressed = event.action == splat::KeyAction::Press;
    if (event.keySym == "w" || event.keySym == "W") {
      flyKeys.forward = pressed;
    } else if (event.keySym == "s" || event.keySym == "S") {
      flyKeys.backward = pressed;
    } else if (event.keySym == "a" || event.keySym == "A") {
      flyKeys.left = pressed;
    } else if (event.keySym == "d" || event.keySym == "D") {
      flyKeys.right = pressed;
    } else if (event.keySym == "q" || event.keySym == "Q") {
      flyKeys.down = pressed;
    } else if (event.keySym == "e" || event.keySym == "E") {
      flyKeys.up = pressed;
    } else if (pressed && (event.keySym == "c" || event.keySym == "C") && hasCameraBounds) {
      placeCameraInsideBounds(visualizer, currentCameraBounds);
    }
  });

  if (!initialFile.empty()) {
    loadFile(visualizer, state, initialFile);
  }

  bool previousO = false;
  bool previousR = false;
  bool previousC = false;
  bool previousOne = false;
  bool previousTwo = false;
  bool visualizerNeedsRedraw = true;
  bool lastRenderOptionsValid = false;
  bool vtkMouseButtonDown = false;
  splat::SplatRenderOptions lastRenderOptions;
  auto lastCameraInteractionTime = std::chrono::steady_clock::time_point::min();
  auto lastSortFreezeInteractionTime = std::chrono::steady_clock::time_point::min();
  auto previousFrameTime = std::chrono::steady_clock::now();
  visualizer.registerMouseCallback([&](const splat::MouseEvent& event) {
    const auto markCameraInteraction = [&](bool freezeSort) {
      const auto interactionTime = std::chrono::steady_clock::now();
      lastCameraInteractionTime = interactionTime;
      visualizerNeedsRedraw = true;
      if (!freezeSort || !state.fastInteraction || !state.sortBackToFront || !visualizer.contains(kCloudId)) {
        return;
      }
      lastSortFreezeInteractionTime = interactionTime;
      auto fastOptions = makeRenderOptions(state);
      fastOptions.freezeSortOrder = true;
      if (!lastRenderOptionsValid || !sameRenderOptions(fastOptions, lastRenderOptions)) {
        visualizer.setSplatRenderOptions(kCloudId, fastOptions);
        lastRenderOptions = fastOptions;
        lastRenderOptionsValid = true;
      }
    };

    const auto applyWheelDolly = [&]() {
      if (event.wheelDelta == 0) {
        return;
      }
      auto* renderer = visualizer.getRenderer();
      if (renderer == nullptr) {
        return;
      }
      auto* camera = renderer->GetActiveCamera();
      const double factor = std::pow(1.1, static_cast<double>(event.wheelDelta));
      camera->Dolly(factor);
      if (hasCameraBounds) {
        setInteriorClippingRange(visualizer, currentCameraBounds);
      }
    };

    if (event.action == splat::MouseAction::Press || event.action == splat::MouseAction::DoubleClick) {
      vtkMouseButtonDown = true;
      markCameraInteraction(true);
    } else if (event.action == splat::MouseAction::Release) {
      vtkMouseButtonDown = false;
      markCameraInteraction(false);
    } else if (event.action == splat::MouseAction::Wheel ||
               (event.action == splat::MouseAction::Move && vtkMouseButtonDown &&
                (event.dx() != 0 || event.dy() != 0))) {
      if (event.action == splat::MouseAction::Wheel) {
        applyWheelDolly();
      }
      markCameraInteraction(event.action != splat::MouseAction::Wheel);
    }
  });

  while (!glfwWindowShouldClose(controlWindow) && !visualizer.wasStopped() && !wantQuit) {
    const auto now = std::chrono::steady_clock::now();
    const double deltaSeconds = std::chrono::duration<double>(now - previousFrameTime).count();
    previousFrameTime = now;

    glfwPollEvents();

    const bool oPressed = glfwGetKey(controlWindow, GLFW_KEY_O) == GLFW_PRESS;
    if (oPressed && !previousO && !ImGui::GetIO().WantCaptureKeyboard) {
      const std::filesystem::path file = openFileDialog();
      if (!file.empty()) {
        visualizerNeedsRedraw = loadFile(visualizer, state, file) || visualizerNeedsRedraw;
        lastRenderOptionsValid = false;
      }
    }
    previousO = oPressed;

    const bool rPressed = glfwGetKey(controlWindow, GLFW_KEY_R) == GLFW_PRESS;
    if (rPressed && !previousR && !ImGui::GetIO().WantCaptureKeyboard) {
      if (hasCameraBounds) {
        resetCameraToBounds(visualizer, currentCameraBounds);
      } else {
        visualizer.resetCamera();
      }
      visualizerNeedsRedraw = true;
    }
    previousR = rPressed;

    const bool cPressed = glfwGetKey(controlWindow, GLFW_KEY_C) == GLFW_PRESS;
    if (cPressed && !previousC && !ImGui::GetIO().WantCaptureKeyboard && hasCameraBounds) {
      placeCameraInsideBounds(visualizer, currentCameraBounds);
      visualizerNeedsRedraw = true;
    }
    previousC = cPressed;

    const bool onePressed = glfwGetKey(controlWindow, GLFW_KEY_1) == GLFW_PRESS;
    if (onePressed && !previousOne && !ImGui::GetIO().WantCaptureKeyboard) {
      state.showStats = !state.showStats;
    }
    previousOne = onePressed;

    const bool twoPressed = glfwGetKey(controlWindow, GLFW_KEY_2) == GLFW_PRESS;
    if (twoPressed && !previousTwo && !ImGui::GetIO().WantCaptureKeyboard) {
      state.showControls = !state.showControls;
    }
    previousTwo = twoPressed;

    visualizer.setBackgroundColor(state.bgR, state.bgG, state.bgB);
    visualizer.setAxesEnabled(state.showAxes);
    if (visualizer.contains(kCloudId)) {
      const bool flyMoved = applyFlyNavigation(visualizer, state, flyKeys, controlWindow, deltaSeconds);
      if (flyMoved) {
        lastCameraInteractionTime = now;
        lastSortFreezeInteractionTime = now;
      }
      visualizerNeedsRedraw = flyMoved || visualizerNeedsRedraw;
      auto renderOptions = makeRenderOptions(state);
      const bool suppressSortForInteraction =
          state.fastInteraction && state.sortBackToFront &&
          lastSortFreezeInteractionTime != std::chrono::steady_clock::time_point::min() &&
          std::chrono::duration<double>(now - lastSortFreezeInteractionTime).count() < kInteractionSortDelaySeconds;
      if (suppressSortForInteraction) {
        renderOptions.freezeSortOrder = true;
      }
      if (!lastRenderOptionsValid || !sameRenderOptions(renderOptions, lastRenderOptions)) {
        visualizer.setSplatRenderOptions(kCloudId, renderOptions);
        lastRenderOptions = renderOptions;
        lastRenderOptionsValid = true;
        visualizerNeedsRedraw = true;
      }
    }
    visualizer.spinOnce(0, visualizerNeedsRedraw);
    visualizerNeedsRedraw = false;

    glfwMakeContextCurrent(controlWindow);
    glViewport(0, 0, controlWindowWidth, controlWindowHeight);
    glClearColor(0.02f, 0.02f, 0.025f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    viewer::renderUIOverlay(state, visualizer.contains(kCloudId));
    if (!visualizer.contains(kCloudId)) {
      viewer::renderOpenPrompt();
    }
    renderStatusToast();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(controlWindow);
  }

  visualizer.close();
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(controlWindow);
  glfwTerminate();
  return 0;
}
