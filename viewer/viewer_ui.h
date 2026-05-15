/**
 * @file viewer_ui.h
 * @brief ImGui UI components for the splat viewer.
 */

#pragma once

#include <filesystem>

namespace viewer {

/**
 * @brief UI state and rendering for the splat viewer.
 */
struct ViewerUIState {
  std::filesystem::path currentFile;            ///< Currently loaded file path
  size_t splatCount = 0;                        ///< Number of loaded splats
  float opacity = 1.0f;                         ///< Global opacity control
  float sizeScale = 1.0f;                       ///< Gaussian size multiplier
  float maxPointSize = 1024.0f;                 ///< Upper bound for projected splat quads
  float flySpeed = 2.0f;                        ///< First-person camera movement speed
  float bgR = 0.05f, bgG = 0.05f, bgB = 0.08f;  ///< Background color
  bool showAxes = true;                         ///< Show VTK axes widget
  bool sortBackToFront = true;                  ///< Sort splats for alpha blending
  bool fastInteraction = true;                  ///< Temporarily relax sorting while camera is moving
  bool clampColors = true;                      ///< Clamp SH DC colors to display range
  bool showStats = true;                        ///< Show statistics panel
  bool showControls = true;                     ///< Show controls panel
};

/**
 * @brief Render the ImGui UI overlay.
 */
void renderUIOverlay(ViewerUIState& state, bool hasFile);

/**
 * @brief Render the file open dialog prompt.
 */
void renderOpenPrompt();

}  // namespace viewer
