/**
 * @file viewer_ui.h
 * @brief ImGui UI components for the splat viewer.
 */

#pragma once

#include <filesystem>
#include <string>

#include "camera.h"
#include "renderer.h"

namespace viewer {

/**
 * @brief UI state and rendering for the splat viewer.
 */
struct ViewerUIState {
  std::filesystem::path currentFile;            ///< Currently loaded file path
  size_t splatCount = 0;                        ///< Number of loaded splats
  float opacity = 1.0f;                         ///< Global opacity control
  float pointSize = 3.0f;                       ///< Point size
  float bgR = 0.05f, bgG = 0.05f, bgB = 0.08f;  ///< Background color
  bool showWireframe = false;                   ///< Wireframe rendering mode
  bool showStats = true;                        ///< Show statistics panel
  bool showControls = true;                     ///< Show controls panel
};

/**
 * @brief Render the ImGui UI overlay.
 */
void renderUIOverlay(ViewerUIState& state, const Camera& camera, const Renderer& renderer, bool hasFile);

/**
 * @brief Render the file open dialog prompt.
 */
void renderOpenPrompt();

}  // namespace viewer
