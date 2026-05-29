# Upstream Reference Hashes

This file pins the upstream reference repositories used for the `ai-superpowers` worktree analysis and implementation.

Date pinned: 2026-05-13

## Repositories

- `reference/engine`
  - Commit: `6df566a371dbd7b33de20a12ad5e05f2465b2a9b`
- `reference/supersplat`
  - Commit: `5b6b8eecec8c7ebf9bbf1c820cba8041a6c50c69`
- `reference/splat-transform`
  - Commit: `bebac611fb1653a701f7f2412d433e89df4f7bf4`

## Purpose

- Keep CLI and data-flow comparisons stable while translating reference behavior into C++.
- Avoid accidental drift if the upstream reference repositories advance during implementation.

---

## splat-transform upstream analysis (bebac61 → 697ea7c, v2.1.1 → v2.4.0)

Analysis date: 2026-05-29

### Version timeline

| Version | Commit | Key changes |
|---------|--------|-------------|
| v2.1.1 | `bebac61` | Pinned baseline |
| v2.2.0 | `c5aa544` | WebP image output via GPU splat rasterizer (#235), filter-sphere CLI fix (#237) |
| v2.2.1 | `b9d9501` | Dependency updates (#242) |
| v2.3.0 | `344702b` | GPU-accelerated KNN and edge cost in decimator (#244), multi-resolution rendering fix (#243) |
| v2.3.1 | `1239a7c` | Dependency updates (#245) |
| v2.3.2 | `05c6e1e` | Dependency updates (#246) |
| v2.4.0 | `3fcb9b4` | Equirectangular projection + shader chunk refactor (#247), DoF (#248), motion blur (#249), dependency updates (#250) |

### Change breakdown by category

#### 1. Algorithmic correctness (MUST follow)

**#241 - Fix decimate color drift: area-weighted, mass-conserving merge math**

The C++ decimate at `src/op/decimate.cpp` + `include/splat/maths/gaussian-decimate-math.h` implements the old NanoGS merge formula. This changed upstream in three ways:

| Aspect | Old (NanoGS, v2.1.1) | New (spark, v2.2.0+) |
|--------|----------------------|----------------------|
| Weight | volume·α: (2π)^1.5·sx·sy·sz·α | area·α: ellipsoidArea(sx,sy,sz)·α |
| Opacity | Porter-Duff: α_i+α_j−α_i·α_j | Mass-conserving: min(1, Σ(α·area)/area_merged) |
| Color | mass-weighted average | area·α weighted normalized average |

The commit message references sparkjsdev/spark `gsplat.rs::new_merged`. The area-weighting better preserves color from thin/anisotropic splats, and the mass-conserving opacity prevents over-saturation. **The C++ code needs this fix.**

**#244 structural improvements to decimate:**

- Float64 → Float32 for per-splat cache (sufficient for MC-sample heuristic, halves memory)
- Remove `Rt` (transpose) from cache — derivable from `R` via index swap
- Pre-allocated `MergeScratch` buffers (eliminates per-call allocation at ~9M merges)
- Remove pre-merge opacity pruning step
- `eigenSymmetric3x3` takes caller-provided scratch instead of allocating internally
- `rotmatToQuat` writes to caller-provided output instead of returning new array

The C++ `gaussian-decimate-math.h` should be audited for the same allocation patterns.

#### 2. New rendering pipeline (new capability, large scope)

**#235 - WebP image output via GPU splat rasterizer** (811 new lines)

Adds a complete GPU compute-shader splat rasterizer. Splats are evaluated per-pixel
(vs. the C++ OpenGL billboard-quad approximation). Pipeline: project → prefix-sum →
tile-bin-emit-pairs → sort → rasterize-binned → finalize. Outputs WebP images.

**#243 - Fix multi-resolution rendering** (key+value sort, sub-frame split, image-relative fade)

**#247 - Equirectangular projection + shader chunk refactor**

Refactored monolithic WGSL into reusable chunks: `constants`, `covariance-3d`,
`jacobian-pinhole`, `jacobian-equirect`, `projection-pinhole`, `projection-equirect`,
`quat-rotation`, `sh-band-1/2/3`, `tile-aabb-*`, `tile-walk-*`.

For C++: This is a major feature addition requiring a CUDA compute path. SplatLib's
current OpenGL billboard renderer is architecturally different. Only worth pursuing
if headless, high-quality image output is a requirement.

#### 3. Post-processing effects (depends on #235/#247)

**#248 - Depth-of-field** (pinhole only, with focusDistance/apertureScale/fStop/sensorSize)
**#249 - Camera motion blur** (sub-frame accumulation with shutter fraction)

#### 4. Camera model (structural gap in C++)

New `src/lib/render/camera.ts` — dedicated `RenderCamera` type with `CameraBasis`:
- Pinhole and equirectangular projection modes
- Camera basis: right/down/forward from position/target/up
- Focal length derivation from FOV

SplatLib currently has no camera model — camera state is scattered across
`GSplatGLFrameState`, VTK's `vtkCamera`, and `CameraController`. Porting this
abstraction would be a clean structural win.

#### 5. CLI options

New `Options` fields: `renderProjection`, `renderCameraPosition`, `renderLookAt`,
`renderUp`, `renderFov`, `renderWidth`, `renderHeight`, `renderNear`, `renderBackground`,
`renderFStop`, `renderFocusDistance`, `renderSensorSize`, `renderCameraEndPosition`,
`renderLookAtEnd`, `renderUpEnd`, `renderShutter`, `renderMotionSamples`.

#### 6. Infrastructure (not applicable to C++)

- `radix-sort.ts` — extracted from decimate into shared spatial module
- `kd-tree.ts` — new GPU-oriented KD-tree alongside existing one
- Dependency updates (5 commits) — npm-only

### Recommendations for C++ follow-up

| Priority | Change | Effort | Rationale |
|----------|--------|--------|-----------|
| **P0** | Port area-weighted merge math (#241) | ~1 day | Bug fix. Old formula causes color drift and over-saturation. Changes are localized to `gaussian-decimate-math.h` and `decimate.cpp`. |
| **P0** | Audit Float64→Float32 in decimate cache | ~0.5 day | Memory savings. Check if C++ already uses f32 or f64. |
| **P1** | Add MergeScratch pre-allocation pattern | ~0.5 day | Avoids per-call heap allocations in merge loop. |
| **P1** | Remove pre-merge opacity pruning | ~0.5 day | Upstream dropped it. Simplifies code. |
| **P2** | Port CameraModel abstraction | ~1-2 days | Structural improvement. Benefits viewers and future headless rendering. |
| **P3** | GPU compute-shader rasterizer | ~2-4 weeks | Major feature. Only if headless WebP/PNG output is needed. Requires CUDA compute path. |
| **P3** | DoF + motion blur | ~1 week | Depends on GPU rasterizer. |
| **P3** | Equirectangular projection | ~1 week | Depends on GPU rasterizer. |
| **N/A** | Shader chunk architecture | — | WGSL-specific. C++ uses GLSL/CUDA. Architectural lesson only. |
| **N/A** | CLI number parsing fix | — | Check if C++ CLI has same `filter-sphere` bug. |
