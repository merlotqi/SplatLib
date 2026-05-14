---
title: Visualization Module Structure
---

# Visualization Module Structure

This document analyzes the visualization-related code found in the historical visualization branch of this repository.

Important note:

- the repository does not currently contain a resolvable `dev-0.3.0` branch or tag
- the closest matching visualization branch present in the repo is `feature/visualization`
- this document is therefore based on `feature/visualization`

## Scope

The visualization work in `feature/visualization` is split into two layers:

1. a reusable library module inside the main `splat` library
2. a standalone viewer application under `viewer/`

That distinction is important:

- `src/visualization/*` and `include/splat/visualization/*` define the reusable visualization API
- `viewer/*` is a separate demo / tool application built on top of the core data model

## File Map

### Reusable visualization module

- `include/splat/visualization/keyevent.h`
- `include/splat/visualization/mouseevent.h`
- `include/splat/visualization/splat_visualizer.h`
- `src/visualization/splat_visualizer.cpp`

### Standalone viewer application

- `viewer/CMakeLists.txt`
- `viewer/README.md`
- `viewer/main.cpp`
- `viewer/camera.h`
- `viewer/camera.cpp`
- `viewer/renderer.h`
- `viewer/renderer.cpp`
- `viewer/shader.h`
- `viewer/viewer_ui.h`
- `viewer/viewer_ui.cpp`

## 1. Build Integration

The visualization module is conditionally compiled into the main library.

Key file:

- `feature/visualization:src/CMakeLists.txt`

Behavior:

- when `BUILD_SPLAT_VISUALIZATION` is enabled:
  - `src/visualization/*.cpp` is added to the `splat` library
  - VTK modules are linked into the core library
- when disabled:
  - the core library builds without visualization support

Main dependencies for the reusable module:

- `VTK::CommonCore`
- `VTK::CommonDataModel`
- `VTK::InteractionStyle`
- `VTK::InteractionWidgets`
- `VTK::RenderingAnnotation`
- `VTK::RenderingCore`
- `VTK::RenderingOpenGL2`

This tells us the intended architecture:

- visualization is not a separate package
- it is an optional extension of the main library
- the API is designed for embedding into host applications

## 2. Public Visualization API

The main public entry point is:

- `include/splat/visualization/splat_visualizer.h`

Core public types:

- `SplatRenderOptions`
- `SplatVisualizer`
- `KeyEvent`
- `MouseEvent`

### `SplatRenderOptions`

This struct defines per-cloud render behavior:

- `globalOpacity`
- `sizeScale`
- `minPointSize`
- `maxPointSize`
- `alphaDiscardThreshold`
- `visible`
- `depthTest`
- `depthWrite`

This is a concise summary of the module's rendering control surface:

- opacity
- splat size scaling
- alpha thresholding
- visibility
- depth behavior

### `SplatVisualizer`

This class is the core reusable visualization object.

Major responsibilities:

- manage a window, renderer, and interactor
- accept `DataTable` splat clouds
- create and remove renderable splat props
- expose camera/render lifecycle controls
- dispatch key and mouse events
- provide optional axes / orientation widget support

Main API groups:

1. Splat cloud lifecycle
- `addSplatCloud`
- `updateSplatCloud`
- `removeSplatCloud`
- `removeAllSplatClouds`
- `contains`
- `getSplatCloudIds`
- `getSplatCount`

2. Per-cloud rendering controls
- `setSplatRenderOptions`
- `multiplySplatSizeScale`

3. Window / scene controls
- `setBackgroundColor`
- `setWindowSize`
- `setWindowName`
- `setAxesEnabled`
- `setAxesLength`
- `resetCamera`

4. Render loop controls
- `render`
- `spin`
- `spinOnce`
- `close`
- `wasStopped`

5. Input callbacks
- `registerKeyCallback`
- `registerMouseCallback`
- clear / unregister variants

6. Raw VTK access
- `getRenderer`
- `getRenderWindow`
- `getInteractor`
- `getAxesWidget`

### API design observation

The public API is intentionally high-level:

- input data type is `DataTable`
- output is a windowed VTK visualization session
- internal rendering details are hidden behind pImpl

That makes `SplatVisualizer` an application-facing facade rather than a low-level renderer object.

## 3. Event System Structure

Key files:

- `include/splat/visualization/keyevent.h`
- `include/splat/visualization/mouseevent.h`

These files define a lightweight event abstraction layer over VTK input events.

### Key events

`KeyEvent` includes:

- `action`
- `modifiers`
- `keySym`
- `keyCode`
- `repeatCount`

Supporting enums:

- `KeyAction`
  - `Press`
  - `Release`
- `KeyModifier`
  - `None`
  - `Shift`
  - `Control`
  - `Alt`

### Mouse events

`MouseEvent` includes:

- `action`
- `button`
- `modifiers`
- `x`, `y`
- `lastX`, `lastY`
- `wheelDelta`
- `repeatCount`

Supporting enums:

- `MouseAction`
  - `Move`
  - `Press`
  - `Release`
  - `DoubleClick`
  - `Wheel`
- `MouseButton`
  - `None`
  - `Left`
  - `Middle`
  - `Right`
  - `Button4`
  - `Button5`

### Architectural role

This layer decouples the public visualization API from raw VTK event constants. That means:

- application code can subscribe without depending on VTK event enums directly
- the visualization module acts as an input translation layer

## 4. Internal Visualization Implementation

The full implementation lives in:

- `src/visualization/splat_visualizer.cpp`

This file is large and effectively contains the entire visualization engine for this branch.

Its structure can be understood as four internal layers.

### A. Data extraction and preprocessing

Early helper functions extract Gaussian attributes from `DataTable`:

- position columns
  - `x`, `y`, `z`
- color columns
  - `f_dc_0`, `f_dc_1`, `f_dc_2`
- scale columns
  - `scale_0`, `scale_1`, `scale_2`
- rotation columns
  - `rot_0`, `rot_1`, `rot_2`, `rot_3`

Important preprocessing steps:

- SH DC is converted to visible RGB using `SH_C0`
- scale is exponentiated from log-scale
- quaternions are normalized
- bounds are computed from positions and Gaussian extents

This is a strong sign that the visualization module consumes the same semantic 3DGS schema used elsewhere in the codebase.

### B. Native OpenGL render prop inside VTK

One of the most important internal classes is:

- `NativeSplatProp : public vtkProp3D`

This class is the real rendering bridge.

Responsibilities:

- integrate custom OpenGL rendering into a VTK render pass
- own GPU-side splat buffers
- provide translucent geometry rendering through VTK
- upload matrices and render uniforms
- sort splats before drawing

Important overridden methods include:

- `RenderOpaqueGeometry`
- `RenderTranslucentPolygonalGeometry`
- `HasTranslucentPolygonalGeometry`
- `ReleaseGraphicsResources`

### C. Custom shader-based splat rendering

The VTK path does not use standard VTK geometry rendering for splats.
Instead it embeds custom GLSL shader code directly in `splat_visualizer.cpp`.

Shader responsibilities:

- vertex shader
  - load position, SH DC color, log-scale, opacity, quaternion
  - compute camera-space covariance
  - estimate projected ellipse / conic
  - compute point size from projected covariance
  - decode color and opacity
- fragment shader
  - evaluate Gaussian falloff in point sprite space
  - discard very low alpha fragments
  - output translucent splat color

This means the visualization module is not just showing points. It is implementing an actual Gaussian splat screen-space approximation.

### D. Visualizer runtime shell

The pImpl-owned runtime state manages:

- `vtkRenderer`
- `vtkRenderWindow`
- `vtkRenderWindowInteractor`
- `vtkInteractorStyleTrackballCamera`
- `vtkAxesActor`
- `vtkOrientationMarkerWidget`
- input observers
- a map of cloud IDs to render props and data

Responsibilities:

- initialize the VTK scene
- attach event observers
- add and remove `NativeSplatProp` instances
- coordinate re-rendering after data changes

## 5. Rendering Data Flow in the Visualization Module

The core flow for the reusable module is:

1. caller provides a `DataTable`
2. `SplatVisualizer::addSplatCloud(...)` clones or stores shared table data
3. a `NativeSplatProp` is created
4. helper functions extract and convert:
   - positions
   - SH DC colors
   - log-scales
   - opacity
   - quaternion rotations
5. GPU buffers are prepared
6. VTK render loop calls `RenderTranslucentPolygonalGeometry`
7. custom GLSL shaders render splats as blended point sprites

This design is notable because it combines:

- VTK for windowing / camera / interaction / scene management
- raw OpenGL for actual splat rendering

## 6. Sort and Transparency Strategy

The implementation is translucent and camera-dependent.

Key signs in the code:

- rendering happens in `RenderTranslucentPolygonalGeometry`
- blending is explicitly enabled
- depth write is configurable and defaults to off in `SplatRenderOptions`
- there is an `UpdateDrawOrder(...)` path tied to model-view state

This suggests the module performs CPU-side draw-order maintenance for correct-ish alpha compositing.

Compared with the later PlayCanvas path, this is simpler:

- no GPU texture stream architecture
- no worker-based sorter
- no tile-based or unified compute pipeline
- direct attribute buffers plus CPU-managed order updates

## 7. Standalone Viewer Structure

The `viewer/` directory is a separate lightweight application.

Key files:

- `viewer/main.cpp`
- `viewer/camera.h`
- `viewer/renderer.h`
- `viewer/renderer.cpp`
- `viewer/viewer_ui.h`
- `viewer/viewer_ui.cpp`
- `viewer/shader.h`

This path is architecturally distinct from the VTK visualization module.

## 8. Viewer Application Architecture

### `viewer/main.cpp`

This is the executable entry point.

Responsibilities:

- initialize GLFW
- create the OpenGL context
- initialize GLEW
- initialize Dear ImGui
- parse file arguments
- load supported splat file formats through `splat::read*`
- compute scene bounds from `x/y/z`
- initialize camera target and radius
- run the main event/render loop

Supported formats:

- `.ply`
- `.splat`
- `.sog`
- `.spz`
- `.ksplat`

### `viewer/camera.h`

Defines a compact orbit camera.

Responsibilities:

- orbit around a target with spherical coordinates
- pan relative to current view
- zoom by adjusting radius
- provide `getViewMatrix()`

This camera is entirely independent of the VTK camera system used by `SplatVisualizer`.

### `viewer/renderer.h/.cpp`

This is a lightweight OpenGL renderer for splats.

Responsibilities:

- extract attributes from `DataTable`
- upload:
  - positions
  - colors
  - scales
  - rotations
- build VAO / VBOs
- render with a simple point-sprite Gaussian shader

Important simplifications compared with `src/visualization/splat_visualizer.cpp`:

- no VTK integration
- simpler shader math
- no explicit covariance-based anisotropic ellipse reconstruction
- no library-grade event abstraction
- intended more as a sample viewer than as an embeddable module

### `viewer/viewer_ui.*`

Handles ImGui overlay state and rendering.

Responsibilities:

- show stats
- show control hints
- show file-open prompts

This is pure application UI and not part of the reusable visualization API.

## 9. Two Visualization Paths in This Branch

This branch effectively contains two visualization implementations:

### A. Library-grade path

- `include/splat/visualization/*`
- `src/visualization/splat_visualizer.cpp`

Characteristics:

- embeddable
- VTK-based interaction shell
- custom OpenGL splat renderer
- callback-based input abstraction

### B. Tool/demo path

- `viewer/*`

Characteristics:

- executable app
- GLFW + GLEW + ImGui
- simpler orbit camera
- simpler OpenGL splat renderer

This is the single most important structural observation for the branch.

## 10. Relationship to the Core Data Model

Both visualization paths consume the core `DataTable` model from the library.

They both assume canonical Gaussian columns such as:

- `x`, `y`, `z`
- `f_dc_0`, `f_dc_1`, `f_dc_2`
- `opacity`
- `scale_0`, `scale_1`, `scale_2`
- `rot_0`, `rot_1`, `rot_2`, `rot_3`

That means the visualization module is not introducing a new scene representation.
Instead it is a rendering layer on top of the existing splat data schema.

## 11. Comparison with Later PlayCanvas-Oriented Work

Compared with the later PlayCanvas references analyzed in `ai/superpowers`, this visualization branch is structurally different:

- it is C++ / VTK / OpenGL native
- it reads `DataTable` directly instead of packing into PlayCanvas texture streams
- it computes rendering data on the CPU and uploads classic GL buffers
- it uses host-framework interaction systems instead of web-engine scene objects

So if PlayCanvas is the reference for web runtime behavior, `feature/visualization` is better understood as:

- a native desktop visualization layer
- a debugging / preview / embedding module
- an earlier or alternate rendering path

## 12. Best Files to Read First

If the goal is to understand the module quickly, read these files in order:

1. `feature/visualization:include/splat/visualization/splat_visualizer.h`
2. `feature/visualization:src/visualization/splat_visualizer.cpp`
3. `feature/visualization:include/splat/visualization/keyevent.h`
4. `feature/visualization:include/splat/visualization/mouseevent.h`
5. `feature/visualization:viewer/README.md`
6. `feature/visualization:viewer/main.cpp`
7. `feature/visualization:viewer/renderer.cpp`

## 13. Bottom-Line Summary

The `feature/visualization` branch adds a real visualization subsystem, not just a sample viewer.

Its structure is:

- public visualization API in headers
- a large VTK-backed visualization implementation in `src/visualization`
- a separate OpenGL + ImGui viewer application in `viewer/`

Architecturally, the reusable module is:

- `DataTable` driven
- VTK managed
- OpenGL rendered
- callback extensible

while the viewer app is:

- lighter weight
- tool oriented
- simpler in rendering and camera logic

