# PlayCanvas Viewer Design

**Date:** 2026-05-15

## Goal

Build a new independent C++ viewer that renders 3D Gaussian splats by following the PlayCanvas TypeScript GSplat rendering path directly, instead of continuing to repair the existing VTK-based `visualization/` and `viewer/` implementation.

The first successful version must load a splat file through public SplatLib readers, convert the column-oriented `DataTable` into a C++ rendering model equivalent to PlayCanvas `GSplatData`, and display visible, depth-sorted Gaussian splats in an OpenGL window.

## Background

The current C++ visualization stack under `include/splat/visualization/`, `src/visualization/`, and `viewer/` has not reached a reliable visible rendering result. The PlayCanvas and SuperSplat TypeScript implementations already render correctly, so the new tool treats the TS path as the correctness baseline.

The important reference files are:

- `reference/engine/src/scene/gsplat/gsplat-data.js`
- `reference/engine/src/scene/gsplat/gsplat-resource.js`
- `reference/engine/src/scene/gsplat/gsplat-instance.js`
- `reference/engine/src/scene/shader-lib/glsl/chunks/gsplat/vert/gsplatCenter.js`
- `reference/engine/src/scene/shader-lib/glsl/chunks/gsplat/vert/gsplatCorner.js`
- `reference/supersplat/src/splat.ts`
- `reference/supersplat/src/shaders/splat-shader.ts`

## Design Decision

Create a new top-level tool directory named `playcanvasViewer/`.

This tool must not depend on the existing VTK `SplatVisualizer`. It may reuse public SplatLib data loading APIs, but it owns its own render data conversion, OpenGL resources, shaders, sorting, camera, and window loop.

The chosen first scope is a PlayCanvas-aligned renderer, not a full SuperSplat editor. It should preserve the important render semantics from PlayCanvas while avoiding selection, editing, state textures, transform palettes, video export, and WebGPU local tile rendering in the first pass.

## Architecture

The new renderer follows this pipeline:

```text
input file
  -> public SplatLib reader
  -> splat::DataTable
  -> PlayCanvas-style GSplat adapter
  -> packed CPU render buffers
  -> OpenGL buffers/textures
  -> CPU back-to-front order update
  -> instanced quad Gaussian shader
  -> GLFW window
```

The C++ adapter intentionally mirrors the uncompressed PlayCanvas path:

- `GSplatData.createIter()` semantics for position, rotation, scale, and color.
- `GSplatResource.updateColorData()` semantics for color and alpha decode.
- `GSplatResource.updateTransformData()` semantics for normalized rotation, exponentiated scale, and packed transform inputs.
- `GSplatInstance.sort()` semantics for reordering by camera position/direction.
- `gsplatCenter` and `gsplatCorner` shader semantics for projected covariance and screen-space ellipse construction.

## Components

### `playcanvasViewer/main.cpp`

Owns CLI parsing, input loading, window creation, and the main loop.

Initial CLI:

```bash
PlaycanvasViewer <input-file>
```

Useful optional flags can be added in the first implementation if they are cheap and local:

- `--width=<pixels>`
- `--height=<pixels>`
- `--max-splats=<count>` for debugging large files
- `--no-sort` for diagnosing ordering issues

The tool should support file types already available from public SplatLib readers: `.ply`, `.splat`, `.sog`, `.spz`, and `.ksplat`.

### `playcanvasViewer/gsplat_data_adapter.h/.cpp`

Converts `splat::DataTable` into a compact render model.

Required input columns:

- `x`, `y`, `z`
- `rot_0`, `rot_1`, `rot_2`, `rot_3`
- `scale_0`, `scale_1`, `scale_2`
- `f_dc_0`, `f_dc_1`, `f_dc_2`
- `opacity`

Decode rules:

- Position is read directly from `x`, `y`, `z`.
- Rotation uses the source layout `(rot_0, rot_1, rot_2, rot_3)` as PlayCanvas `(w, x, y, z)`, normalizes it, and flips sign if `w < 0`, matching `GSplatResource.updateTransformData()`.
- Scale is `exp(scale_i)`.
- Color is `0.5 + f_dc_i * 0.28209479177387814`.
- Alpha is `sigmoid(opacity)`.
- Invalid quaternion values fall back to identity rotation.
- Missing required columns are fatal errors with the missing column name.

The adapter output should be a plain C++ struct with vectors for centers, rotations, scales, colors, and original indices. This keeps the renderer independent from `DataTable` and makes adapter semantics testable without an OpenGL context.

### `playcanvasViewer/gsplat_renderer.h/.cpp`

Owns OpenGL resources and draw submission.

Responsibilities:

- Build static quad corner geometry for instanced rendering.
- Upload decoded per-splat center, rotation, scale, and color buffers.
- Maintain a CPU order vector sorted back-to-front by camera-space depth.
- Upload the current order buffer when the camera changes enough to require re-sorting.
- Draw all visible splats as instanced quads using premultiplied alpha blending.

The first version may use shader storage buffers or vertex attributes depending on local OpenGL availability. The implementation should choose the simplest path supported by the repo's current GLFW/GLEW/OpenGL setup. Texture packing identical to PlayCanvas is desirable, but not required for the first visible closure if buffer attributes preserve the same decoded values and shader math.

### `playcanvasViewer/gsplat_shader_sources.h`

Contains GLSL shader strings translated from PlayCanvas chunks.

Required shader behavior:

- Project the Gaussian center into view and clip space.
- Cull splats behind the camera in perspective mode.
- Build a 3D covariance matrix from normalized rotation and decoded scale.
- Project covariance to screen space with the PlayCanvas Jacobian logic.
- Compute ellipse axes from covariance eigenvalues.
- Emit quad corners in clip space.
- Apply Gaussian falloff in the fragment shader using the PlayCanvas normalized exponential form.
- Premultiply RGB by alpha in the fragment output.

First version may ignore:

- spherical harmonics beyond DC color
- selection and locked state
- outline/ring modes
- transform palette
- pick pass
- WebGPU-specific branches
- fisheye projection

### `playcanvasViewer/camera_controller.h/.cpp`

Provides a small camera controller independent from VTK.

Minimum controls:

- mouse drag: orbit around current focal point
- mouse wheel: dolly
- `W/A/S/D/Q/E`: fly camera
- `R`: reset camera to data bounds
- `Esc`: close

Camera reset uses the adapted centers and decoded scales to compute a useful scene bound. The initial view should place the camera outside the bound and look at the scene center.

## Build Integration

Add a top-level option:

```cmake
option(BUILD_SPLAT_PLAYCANVAS_VIEWER "Build PlayCanvas-style OpenGL splat viewer" OFF)
```

When enabled, add `playcanvasViewer/` as a subdirectory.

The executable should link:

- `SPLAT::splat`
- `glfw`
- `GLEW::GLEW`
- `OpenGL::GL`

It should not require VTK.

## Validation Strategy

### Adapter Tests

Add focused tests that construct a tiny `DataTable` and verify decoded output:

- DC color follows `0.5 + f_dc * SH_C0`.
- Opacity follows sigmoid.
- Scale follows exponentiation.
- Quaternion is normalized and sign-adjusted when `w < 0`.
- Missing columns fail with a direct error.

### Build Validation

Configure and build with:

```bash
cmake -S . -B build/playcanvas-viewer -DBUILD_SPLAT_PLAYCANVAS_VIEWER=ON
cmake --build build/playcanvas-viewer --target PlaycanvasViewer -j
```

Expected result: the executable builds without enabling `BUILD_SPLAT_VISUALIZATION` or VTK.

### Runtime Validation

Run the viewer against a real local input file:

```bash
./build/playcanvas-viewer/playcanvasViewer/PlaycanvasViewer <input-file>
```

Expected result: a window opens with visible Gaussian splats, camera controls work, and the background does not remain blank for valid input.

If the local environment supports offscreen OpenGL reliably, add a smoke path that renders one frame and checks that at least one pixel differs from the clear color. This is useful but not required for the first implementation because CI/headless GL availability may vary.

## Non-Goals For First Pass

- Do not modify or remove the current VTK visualizer.
- Do not implement SuperSplat editing tools.
- Do not implement selection, deletion, locking, state textures, or transform palettes.
- Do not implement SH bands beyond DC color.
- Do not implement PlayCanvas WebGPU local tile rendering.
- Do not implement LOD streaming in the viewer.
- Do not make this a general replacement for `viewer/` until the new path is visibly verified.

## Risks And Mitigations

### Risk: Diverging From PlayCanvas Render Math

Mitigation: keep shader source organization close to the reference chunks and name helper functions after the PlayCanvas concepts: center, corner, covariance, and falloff.

### Risk: First Implementation Becomes Another Monolith

Mitigation: keep data adaptation, render submission, shader strings, and camera control in separate files. The adapter must be testable without OpenGL.

### Risk: CPU Sorting Is Too Slow For Very Large Files

Mitigation: accept CPU sorting for the first visible closure. Add `--max-splats` and `--no-sort` as debug aids. GPU sorting can be a later optimization once correctness is established.

### Risk: OpenGL Feature Availability Differs Across Machines

Mitigation: start with OpenGL 3.3-friendly instanced vertex attributes if possible. Use shader storage buffers only if the local dependency stack and target machines support them cleanly.

## Success Criteria

The first implementation is successful when:

1. `PlaycanvasViewer` builds independently of VTK.
2. A small adapter test confirms PlayCanvas decode semantics from `DataTable`.
3. A real splat file opens in a GLFW window.
4. The rendered output is visibly non-empty and uses Gaussian elliptical splats rather than point sprites.
5. Camera movement and reset work well enough to inspect the scene.
