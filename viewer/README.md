# Splat Viewer

An ImGui control shell for the VTK-based `splat::SplatVisualizer`.

The viewer intentionally does not implement a second splat renderer. It loads splat files into the shared
`SplatCloud` model, forwards them to `SplatVisualizer`, and uses ImGui for runtime controls.

## Features

- Multi-format support: PLY, `.splat`, SOG, SPZ, and KSplat
- VTK rendering through `splat::SplatVisualizer`
- Dear ImGui control window for render options and status
- Shared visualization path with the C++ library API

## Building

Enable the viewer in your CMake configuration:

```bash
cmake .. -DBUILD_SPLAT_VIEWER=ON
cmake --build . --target splat_viewer
```

`BUILD_SPLAT_VIEWER=ON` automatically enables `BUILD_SPLAT_VISUALIZATION=ON`.

## Dependencies

- VTK: rendering, interaction style, widgets, and OpenGL2 backend
- GLFW: ImGui control window
- GLEW and OpenGL: ImGui OpenGL backend
- Dear ImGui 1.90.9: fetched by CMake

## Usage

```bash
splat_viewer scene.ply
splat_viewer
```

The VTK render window handles camera interaction. The ImGui control window handles loading and render options.

| Action | Input |
|--------|-------|
| Open file | O key in the control window |
| Reset camera | R key in the control window |
| Toggle stats | 1 key in the control window |
| Toggle controls | 2 key in the control window |
| Rotate / pan / zoom | Use the VTK render window |

## Architecture

```text
viewer/
├── CMakeLists.txt      # ImGui control app target
├── main.cpp            # File loading, app loop, SplatVisualizer integration
├── viewer_ui.h         # ImGui state and UI declarations
└── viewer_ui.cpp       # ImGui panels
```

Rendering lives in `SplatVisualizer`, `SplatGaussianProp`, and the shared `GSplatGLRenderer`, so viewer behavior stays
aligned with the library visualization module.
