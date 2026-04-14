# Splat Viewer

An OpenGL + Dear ImGui viewer for 3D Gaussian Splatting data.

## Features

- **Multi-format support**: PLY, .splat, SOG, SPZ, KSplat, and LCC
- **Interactive camera controls**: Orbit, pan, zoom
- **ImGui overlay**: Statistics and rendering controls
- **Cross-platform**: Windows, macOS, Linux

## Building

Enable the viewer in your CMake configuration:

```bash
cmake .. -DBUILD_SPLAT_VIEWER=ON
cmake --build . --target splat_viewer
```

### Dependencies (automatically fetched)

- **GLFW 3.3** - Window management
- **GLAD 0.1** - OpenGL loader
- **Dear ImGui 1.90** - UI framework
- **GLM 1.0** - Mathematics library

## Usage

### Launch

```bash
# Open with file argument
splat_viewer scene.ply

# Open without file (press O to browse)
splat_viewer
```

### Controls

| Action | Input |
|--------|-------|
| Rotate | Left mouse drag |
| Pan | Right mouse drag |
| Zoom | Scroll wheel |
| Open file | O key |
| Reset camera | R key |
| Toggle stats | 1 key |
| Toggle controls | 2 key |

## Architecture

```
viewer/
├── CMakeLists.txt      # Build configuration with FetchContent
├── main.cpp            # Application entry point
├── camera.h/cpp        # Orbit camera implementation
├── renderer.h/cpp      # OpenGL splat renderer
├── viewer_ui.h/cpp     # ImGui UI components
└── shader.h            # Shader compilation helper
```

## Rendering Pipeline

1. Load DataTable from file (PLY, .splat, etc.)
2. Extract position, color, scale, rotation columns
3. Upload to GPU vertex buffers
4. Render as point sprites with Gaussian falloff shader
5. ImGui overlay for controls and statistics

## Notes

- Colors are derived from SH DC coefficients: `color = f_dc * 0.282 + 0.5`
- Scales are stored in log space: `scale = exp(log_scale)`
- Quaternions are normalized on load
- Point size is dynamically adjusted based on depth
