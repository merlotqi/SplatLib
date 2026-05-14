---
title: PlayCanvas 3DGS Code Map
---

# PlayCanvas 3DGS Code Map

This document maps the PlayCanvas 3D Gaussian Splatting code pulled into the `ai/superpowers` worktree through:

- `reference/engine`
- `reference/supersplat`

It focuses on three things:

1. Rendering logic
2. 3DGS data structure definitions
3. Shader locations and responsibilities

## Upstream Repositories

- `reference/engine`
  - Role: low-level runtime for gsplat loading, resource creation, sorting, and rendering
- `reference/supersplat`
  - Role: editor and tooling layer built on top of PlayCanvas gsplat runtime

## High-Level Split

The code is split cleanly across two layers.

### `reference/engine`

This is the runtime layer. It owns:

- gsplat asset/resource loading
- in-memory gsplat data containers
- GPU resource setup
- sort order generation
- core gsplat shader chunks
- unified gsplat rendering pipeline

### `reference/supersplat`

This is the editor/application layer. It owns:

- file loading through `@playcanvas/splat-transform`
- creation of editable `Splat` objects
- per-splat edit state such as selected, deleted, locked
- per-splat transform palette indirection
- picking, overlays, bounds, and editing shaders

## `reference/engine` Code Map

### 1. Asset Loading and Resource Construction

Key files:

- `reference/engine/src/framework/handlers/gsplat.js`
- `reference/engine/src/framework/components/gsplat/gsplat-asset-loader.js`
- `reference/engine/src/scene/gsplat/gsplat-resource-base.js`
- `reference/engine/src/scene/gsplat/gsplat-resource.js`
- `reference/engine/src/scene/gsplat/gsplat-compressed-resource.js`
- `reference/engine/src/scene/gsplat/gsplat-sog-resource.js`

Responsibilities:

- recognize and load gsplat assets
- build `GSplatResource*` objects from parsed data
- allocate textures / streams used by render-time shaders
- expose mesh, centers, chunks, SH data, and format-specific GPU views

`GSplatResourceBase` is the common root for renderable gsplat resources. The more specific resource classes specialize for uncompressed, compressed, and SOG-backed data.

### 2. 3DGS Data Structures

Key files:

- `reference/engine/src/scene/gsplat/gsplat-data.js`
- `reference/engine/src/scene/gsplat/gsplat-compressed-data.js`
- `reference/engine/src/scene/gsplat/gsplat-sog-data.js`
- `reference/engine/src/scene/gsplat/gsplat-format.js`
- `reference/engine/src/scene/gsplat/gsplat-streams.js`
- `reference/engine/src/scene/gsplat/gsplat-resolve-sh.js`

Responsibilities:

- define how splat vertex properties are stored and accessed
- represent uncompressed, compressed, and SOG variants
- convert raw property storage into render-friendly streams
- resolve spherical harmonics data for runtime shading

Important details from `gsplat-data.js`:

- `GSplatData` stores elements and comments from parsed data
- the main element is `vertex`
- canonical per-splat properties include:
  - position: `x`, `y`, `z`
  - rotation quaternion: `rot_0`, `rot_1`, `rot_2`, `rot_3`
  - scale: `scale_0`, `scale_1`, `scale_2`
  - base color / SH DC: `f_dc_0`, `f_dc_1`, `f_dc_2`
  - opacity: `opacity`
- `createIter()` exposes a splat iterator that reconstructs:
  - position
  - normalized rotation
  - exponentiated scale
  - decoded color and opacity

This file is the clearest reference point for the PlayCanvas-side semantic meaning of 3DGS attributes.

### 3. Render Instance and Sort Pipeline

Key files:

- `reference/engine/src/scene/gsplat/gsplat-instance.js`
- `reference/engine/src/scene/gsplat/gsplat-sorter.js`
- `reference/engine/src/scene/gsplat/gsplat-sort-worker.js`
- `reference/engine/src/scene/gsplat/gsplat-container.js`

Responsibilities:

- create per-instance render state
- own the material bound to a gsplat draw
- maintain sort buffers / order textures
- update draw order when the camera changes

`gsplat-instance.js` is the main runtime entry point for a rendered splat object:

- creates an order target
  - `StorageBuffer` on WebGPU
  - `Texture` on WebGL
- creates a `ShaderMaterial` using gsplat shader chunks
- builds a `MeshInstance`
- creates a `GSplatSorter` when center data is available
- updates sorter camera state in `sort(cameraNode)`

This is the core file to read when tracing "how a loaded gsplat becomes an actual draw call".

### 4. Shader Locations

Core gsplat shader chunks live here:

- GLSL vertex:
  - `reference/engine/src/scene/shader-lib/glsl/chunks/gsplat/vert/*`
- GLSL fragment:
  - `reference/engine/src/scene/shader-lib/glsl/chunks/gsplat/frag/*`
- WGSL:
  - `reference/engine/src/scene/shader-lib/wgsl/chunks/gsplat/*`

Important GLSL vertex chunks:

- `gsplat.js`
- `gsplatCommon.js`
- `gsplatStructs.js`
- `gsplatSource.js`
- `gsplatFormat.js`
- `gsplatEvalSH.js`
- `gsplatSplat.js`
- `gsplatOutput.js`
- `formats/uncompressed.js`
- `formats/compressed.js`
- `formats/sog.js`

Important fragment chunks:

- `frag/gsplat.js`
- `frag/gsplatProcess.js`
- `frag/gsplatPacking.js`

What these chunks cover:

- reading splat data from textures / streams
- format-specific unpacking
- SH evaluation
- quad / ellipse generation logic
- color and alpha processing
- intermediate packing for work buffers or composite passes

### 5. Unified GSplat Pipeline

Key directory:

- `reference/engine/src/scene/gsplat-unified/`

Representative files:

- `gsplat-manager.js`
- `gsplat-renderer.js`
- `gsplat-projector.js`
- `gsplat-frustum-culler.js`
- `gsplat-unified-sorter.js`
- `gsplat-hybrid-renderer.js`
- `gsplat-budget-balancer.js`
- `gsplat-work-buffer.js`
- `gsplat-tile-composite.js`
- `frame-pass-gsplat-compute-local.js`

This directory contains the newer, more modular pipeline for gsplat rendering. Compared with the classic `gsplat-instance` path, it adds:

- compute-heavy projection and placement stages
- tile-based or work-buffer-based passes
- frustum culling and budget balancing
- local / hybrid rendering strategies
- a more explicit renderer-director-manager split

If the goal is to understand the latest PlayCanvas gsplat rendering architecture, this is the most important directory after the basic `scene/gsplat` runtime.

## `reference/supersplat` Code Map

### 1. Asset Loading Entry Point

Key file:

- `reference/supersplat/src/asset-loader.ts`

Responsibilities:

- load source files through `@playcanvas/splat-transform`
- validate and normalize returned gsplat data
- create a PlayCanvas `Asset`
- assign a `GSplatResource`
- wrap the asset in an editor-facing `Splat` object

This is the bridge from file I/O into the PlayCanvas runtime.

### 2. Editor-Side `Splat` Object

Key file:

- `reference/supersplat/src/splat.ts`

Responsibilities:

- represent one editable splat scene object
- attach a PlayCanvas `gsplat` component to an `Entity`
- extend raw gsplat data with editor-only properties
- push selection / deletion / transform state to GPU textures
- trigger resorting and bound recomputation after edits

Important editor-added properties:

- `state`
  - `uchar` per splat
  - bitfield used for selected / deleted / locked state
- `transform`
  - `ushort` per splat
  - index into a transform palette texture / buffer

Important GPU state created by `Splat`:

- `stateTexture`
  - `PIXELFORMAT_R8`
  - stores per-splat state bits
- `transformTexture`
  - `PIXELFORMAT_R16U`
  - stores transform palette indices
- `transformPalette`
  - stores actual transform matrices used by selected splats

Important runtime hooks:

- `rebuildMaterial()`
  - overrides engine gsplat shader chunks with supersplat shader code
- `updateState()`
  - uploads selection / deletion / lock state
- `updatePositions()`
  - updates sorter centers after transforms
- `updateSorting()`
  - rebuilds sorter mapping when deleted splats should be removed from rendering

This file is the main extension point where supersplat turns a render-only gsplat into an editable object.

### 3. Scene, Camera, Rendering, and Picking

Key files:

- `reference/supersplat/src/scene.ts`
- `reference/supersplat/src/scene-state.ts`
- `reference/supersplat/src/camera.ts`
- `reference/supersplat/src/render.ts`
- `reference/supersplat/src/picker.ts`

Responsibilities:

- editor scene setup
- camera control and user navigation
- render orchestration and frame updates
- ID and depth based picking for splats

`picker.ts` is especially important for editor behavior because it prepares temporary passes that render a specific splat for:

- ID picking
- rectangle selection
- depth picking

### 4. GPU-Assisted Editing Helpers

Key files:

- `reference/supersplat/src/splats-transform-handler.ts`
- `reference/supersplat/src/splat-overlay.ts`

Responsibilities:

- apply transforms to selected splats through palette indirection
- hide editor latency by updating GPU-side state first
- render overlay points for selected / editable splats
- read and reuse engine order / position / color / SH textures

These files are where supersplat starts to look like a GPU-assisted editor instead of a simple viewer.

## Supersplat Shader Map

Key files:

- `reference/supersplat/src/shaders/splat-shader.ts`
- `reference/supersplat/src/shaders/splat-overlay-shader.ts`
- `reference/supersplat/src/shaders/position-shader.ts`
- `reference/supersplat/src/shaders/bound-shader.ts`
- `reference/supersplat/src/shaders/intersection-shader.ts`

### `splat-shader.ts`

Purpose:

- overrides the default engine gsplat shader path for editor use
- reads `splatState` and `splatTransform`
- filters deleted / locked / selected splats based on operation mode
- supports pick/depth/editor display modes
- applies tint, saturation, brightness, and other editor display adjustments

This is the most important supersplat shader file.

### `splat-overlay-shader.ts`

Purpose:

- renders per-splat overlay points on top of the normal splat image
- uses the sorted order texture to match visible render order
- can optionally reconstruct gaussian color from SH data
- highlights selected / unselected state

### `position-shader.ts`

Purpose:

- reconstructs splat center positions from engine textures
- applies transform palette entries
- supports GPU-side position extraction for downstream tools

### `bound-shader.ts`

Purpose:

- computes selected and visible bounds
- ignores deleted splats
- can incorporate per-splat transforms

### `intersection-shader.ts`

Purpose:

- supports editor intersection tests against transformed splat centers

## End-to-End Render Flow

The practical render chain across the two repositories is:

1. A source file is loaded by `reference/supersplat/src/asset-loader.ts`
2. The loader produces `GSplatData` and wraps it in `GSplatResource`
3. `reference/supersplat/src/splat.ts` creates an `Entity` with a `gsplat` component
4. PlayCanvas runtime creates a `GSplatInstance`
5. `GSplatInstance` allocates order buffers / textures and a gsplat material
6. `GSplatSorter` updates visible draw order from camera motion
7. Engine gsplat shader chunks read stream / texture data and render splats
8. Supersplat may override the shader path to inject editor-only state and overlays

For editable splats, the chain becomes:

1. raw gsplat data
2. add `state` and `transform` properties
3. upload state / transform indices to textures
4. update transform palette
5. update sorter centers or mapping when selection / deletion changes
6. re-render through supersplat shader overrides

## Best Starting Points for Further Reading

If you only want the minimum set of files to understand PlayCanvas 3DGS behavior, start here:

1. `reference/engine/src/scene/gsplat/gsplat-data.js`
2. `reference/engine/src/scene/gsplat/gsplat-resource.js`
3. `reference/engine/src/scene/gsplat/gsplat-instance.js`
4. `reference/engine/src/scene/shader-lib/glsl/chunks/gsplat/vert/gsplat.js`
5. `reference/supersplat/src/asset-loader.ts`
6. `reference/supersplat/src/splat.ts`
7. `reference/supersplat/src/shaders/splat-shader.ts`

## Relevance to `ai/superpowers`

These references matter because the `ai/superpowers` worktree is using PlayCanvas behavior as a source of truth for:

- PlayCanvas-oriented 3DGS data interpretation
- runtime/rendering expectations for PLY, compressed splat data, and SOG
- editor-visible per-splat state and transform semantics
- future alignment between SplatLib processing code and PlayCanvas rendering behavior
