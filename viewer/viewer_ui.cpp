/**
 * @file viewer_ui.cpp
 * @brief ImGui UI implementation for the splat viewer.
 */

#include "viewer_ui.h"

#include <imgui.h>

namespace viewer {

/**
 * @brief Render the ImGui UI overlay.
 *
 * @param state UI state
 * @param hasFile Whether a file is loaded
 */
void renderUIOverlay(ViewerUIState& state, bool hasFile) {
  // Stats panel
  if (state.showStats) {
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::SetNextWindowSize(ImVec2(300, 96));
    ImGui::Begin("Statistics", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);

    if (hasFile) {
      const std::string displayName = state.currentFile.filename().u8string();
      ImGui::Text("File: %s", displayName.c_str());
      ImGui::Text("Splats: %zu", state.splatCount);
    } else {
      ImGui::Text("No file loaded");
    }

    ImGui::End();
  }

  // Controls panel
  if (state.showControls) {
    ImGui::SetNextWindowPos(ImVec2(10, 116));
    ImGui::SetNextWindowSize(ImVec2(300, 286));
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoResize);

    ImGui::TextUnformatted("Rendering");
    ImGui::SliderFloat("Opacity", &state.opacity, 0.1f, 1.0f);
    ImGui::SliderFloat("Size Scale", &state.sizeScale, 0.1f, 10.0f);
    ImGui::SliderFloat("Max Point Size", &state.maxPointSize, 32.0f, 1024.0f);
    ImGui::SliderFloat("Fly Speed", &state.flySpeed, 0.05f, 20.0f);
    ImGui::Checkbox("Sort Back To Front", &state.sortBackToFront);
    ImGui::Checkbox("Fast Interaction", &state.fastInteraction);
    ImGui::Checkbox("Clamp Colors", &state.clampColors);
    ImGui::Checkbox("Axes", &state.showAxes);

    ImGui::TextUnformatted("Background Color");
    ImGui::ColorEdit3("Color", &state.bgR, ImGuiColorEditFlags_NoInputs);

    ImGui::TextUnformatted("");
    ImGui::TextDisabled("VTK window handles camera interaction");
    ImGui::TextDisabled("O: open | R: reset | C: center inside");
    ImGui::TextDisabled("WASD/QE: fly camera through rooms");

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
