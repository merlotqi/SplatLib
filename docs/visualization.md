# Visualization

SplatLib routes Gaussian splat rendering through VTK when `BUILD_SPLAT_VISUALIZATION` is enabled. The recommended integration point is `splat::SplatVisualizer`: it owns or attaches to VTK rendering objects, keeps mouse and camera interaction inside VTK, and exposes the renderer/window/interactor needed by Qt, ImGui, and custom viewers.

Qt applications can place the visualizer's `vtkRenderWindow` in `QVTKOpenGLNativeWidget` or `QVTKRenderWidget`. ImGui or GLFW shells should treat VTK as the rendering canvas and use ImGui only for controls.

## VTK Canvas Integration

Use `splat::SplatVisualizer` when your application owns the UI shell but wants VTK to own rendering, camera, and mouse interaction.

```cpp
#include <splat/splat.h>
#include <splat/visualization/splat_visualizer.h>

#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>

auto data = splat::readPly("scene.ply");

auto renderer = vtkSmartPointer<vtkRenderer>::New();
auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();

splat::SplatVisualizer viewer(renderer, renderWindow, interactor, "Splat Viewer");

splat::SplatRenderOptions options;
options.sortBackToFront = true;
options.depthTest = true;
options.depthWrite = false;
options.sizeScale = 1.0f;
viewer.addSplatCloud(std::move(data), "scene", options);
viewer.resetCamera();

// Qt: widget->setRenderWindow(viewer.getRenderWindow());
// ImGui/GLFW shells: keep VTK's render window as the 3D canvas and use ImGui for controls.
```

`SplatVisualizer` wraps `vtkRenderer`, `vtkRenderWindow`, `vtkRenderWindowInteractor`, camera interaction, axes, and one or more `SplatGaussianProp` instances.

## Managed Window

Use `splat::SplatVisualizer` when you want SplatLib to own a simple VTK window and interactor.

```cpp
#include <splat/splat.h>
#include <splat/visualization/splat_visualizer.h>

auto data = splat::readPly("scene.ply");

splat::SplatVisualizer viewer("Splat Viewer");
viewer.addSplatCloud(std::move(data));
viewer.resetCamera();
viewer.spin();
```

## Low-Level Prop

Use `splat::SplatGaussianProp` only when your application already has a complete VTK renderer/window/interactor setup and wants to manage camera and input itself.

```cpp
#include <splat/splat.h>
#include <splat/visualization/splat_gaussian_prop.h>

#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

auto data = splat::readPly("scene.ply");

auto prop = vtkSmartPointer<splat::SplatGaussianProp>::New();
prop->SetInputData(std::move(data));

vtkRenderer* renderer = /* owned by your application */;
renderer->AddViewProp(prop);
renderer->ResetCamera();
```

`SplatGaussianProp` accepts the same `SplatCloud` columns used by PlayCanvas-style Gaussian splats: `x`, `y`, `z`, `rot_0..3`, `scale_0..2`, `f_dc_0..2`, and `opacity`. Internally it converts `SplatCloud` to `GSplatData` and renders through the shared PlayCanvas-compatible OpenGL path.

## Build

```bash
cmake -S . -B build -DBUILD_SPLAT_VISUALIZATION=ON
cmake --build build --target splat
```

When using the optional standalone PlayCanvas-style viewer, enable `BUILD_SPLAT_PLAYCANVAS_VIEWER` as well.
