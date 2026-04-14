/**
 * @file viewer_ui.cpp
 * @brief ImGui UI implementation for the splat viewer.
 */

#include "viewer_ui.h"

#include <imgui.h>

#include <iomanip>
#include <sstream>

#include "camera.h"
#include "renderer.h"

namespace viewer {

/**
 * @brief Render the ImGui UI overlay.
 *
 * @param state UI state
 * @param camera Camera for display
 * @param renderer Renderer for splat count info
 * @param hasFile Whether a file is loaded
 */
void renderUIOverlay(ViewerUIState& state, const Camera& camera, const Renderer& renderer, bool hasFile) {
  // Stats panel
  if (state.showStats) {
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::SetNextWindowSize(ImVec2(260, 120));
    ImGui::Begin("Statistics", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);

    if (hasFile) {
      // Extract filename from path
      std::string displayName = state.currentFile;
      size_t lastSlash = displayName.find_last_of("/\\");
      if (lastSlash != std::string::npos) {
        displayName = displayName.substr(lastSlash + 1);
      }
      ImGui::Text("File: %s", displayName.c_str());
      ImGui::Text("Splats: %zu", renderer.getSplatCount());
    } else {
      ImGui::Text("No file loaded");
    }

    ImGui::Text("Camera: (%.2f, %.2f, %.2f)", camera.getPosition().x, camera.getPosition().y, camera.getPosition().z);
    ImGui::Text("Distance: %.2f", camera.radius);

    ImGui::End();
  }

  // Controls panel
  if (state.showControls) {
    ImGui::SetNextWindowPos(ImVec2(10, 140));
    ImGui::SetNextWindowSize(ImVec2(260, 200));
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoResize);

    ImGui::TextUnformatted("Rendering");
    ImGui::SliderFloat("Opacity", &state.opacity, 0.1f, 1.0f);
    ImGui::SliderFloat("Point Size", &state.pointSize, 1.0f, 10.0f);
    ImGui::Checkbox("Wireframe", &state.showWireframe);

    ImGui::TextUnformatted("Background Color");
    ImGui::ColorEdit3("Color", &state.bgR, ImGuiColorEditFlags_NoInputs);

    ImGui::TextUnformatted("Camera");
    ImGui::SliderFloat("Radius", &const_cast<Camera&>(camera).radius, 1.0f, 1000.0f);

    ImGui::TextUnformatted("");
    ImGui::TextDisabled("Left-drag: Rotate | Right-drag: Pan");
    ImGui::TextDisabled("Scroll: Zoom");

    ImGui::End();
  }
}

/**
 * @brief Render the file open dialog prompt.
 */
void renderOpenPrompt() {
  ImGui::SetNextWindowPos(
      ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f - 150.0f, ImGui::GetIO().DisplaySize.y * 0.5f - 30.0f));
  ImGui::Begin("Drop File", nullptr,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysAutoResize |
                   ImGuiWindowFlags_NoMove);
  ImGui::TextUnformatted("Press O to open file");
  ImGui::End();
}

}  // namespace viewer
