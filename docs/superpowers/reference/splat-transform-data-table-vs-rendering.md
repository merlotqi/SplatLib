---
title: Splat-Transform DataTable vs PlayCanvas Rendering Structures
---

# Splat-Transform DataTable vs PlayCanvas Rendering Structures

This document compares:

- `reference/splat-transform/src/lib/data-table/data-table.ts`
- PlayCanvas runtime rendering structures in `reference/engine`
- editor-side rendering extensions in `reference/supersplat`

The goal is to clarify how the same 3DGS scene data changes shape as it moves from:

1. a general-purpose transform / processing table
2. a PlayCanvas runtime data object
3. GPU-oriented render textures and sort buffers
4. supersplat editor-only per-splat state

## Short Version

The core difference is:

- `splat-transform` uses a generic CPU-side column table
- PlayCanvas rendering uses a more specialized splat object model plus GPU texture streams

In other words:

- `DataTable` is a flexible processing representation
- `GSplatData` is a PlayCanvas-oriented semantic representation
- `GSplatResource` is a GPU-ready packed representation
- supersplat adds editor-only rendering state on top

## 1. `splat-transform` `DataTable`

Primary files:

- `reference/splat-transform/src/lib/data-table/data-table.ts`
- `reference/splat-transform/src/lib/data-table/sh-bands.ts`
- `reference/splat-transform/src/lib/data-table/transform.ts`

### Core model

`DataTable` is a generic columnar container:

- `Column`
  - `{ name, data }`
- `data`
  - one typed array per column
- `DataTable`
  - `columns: Column[]`
  - `transform: Transform`

### Important properties

- all columns must have the same row count
- column names are the semantic contract
- column storage is typed-array based
- the table is intentionally format-agnostic and processing-oriented

### Standard Gaussian columns

The expected 3DGS columns are:

- position
  - `x`, `y`, `z`
- rotation quaternion
  - `rot_0`, `rot_1`, `rot_2`, `rot_3`
- scale
  - `scale_0`, `scale_1`, `scale_2`
- color DC
  - `f_dc_0`, `f_dc_1`, `f_dc_2`
- opacity
  - `opacity`
- higher-order SH
  - `f_rest_0` through `f_rest_44`

### Important semantic detail

In `DataTable`, the raw values are still in processing form:

- scale columns are log-scale
- opacity is logit
- SH bands are inferred from column presence
- coordinate space is tracked separately in `dataTable.transform`

That last point matters a lot: `DataTable` separates:

- raw numeric arrays
- coordinate-space metadata

This makes it suitable for format conversion and preprocessing.

## 2. PlayCanvas `GSplatData`

Primary file:

- `reference/engine/src/scene/gsplat/gsplat-data.js`

### Core model

`GSplatData` is still CPU-side, but it is no longer a generic processing table.
It is a PlayCanvas splat data object built around parsed PLY-style elements:

- `elements`
- `comments`
- `numSplats`

The main data still lives in named properties under the `vertex` element, and the canonical names match the `DataTable` convention:

- `x`, `y`, `z`
- `rot_0` .. `rot_3`
- `scale_0` .. `scale_2`
- `f_dc_0` .. `f_dc_2`
- `opacity`
- `f_rest_*`

### Similarity to `DataTable`

The field naming and row-wise meaning are largely the same.

This is the most important compatibility point:

- both systems use the same canonical Gaussian property names
- both systems store one value per splat per property
- both systems treat SH columns as channel-major `f_rest_*`

### Difference from `DataTable`

`GSplatData` is less generic and more render-semantic:

- it is built around PLY element/property shape instead of a standalone table abstraction
- it does not carry a separate `transform` metadata object like `DataTable`
- it exposes render-facing helpers such as:
  - `createIter()`
  - `calcAabb()`
  - `calcAabbExact()`

### Important semantic detail

`GSplatData.createIter()` starts decoding values into render-meaningful form:

- scale is converted with `exp()`
- opacity is converted with `sigmoid()`
- DC color is converted with the SH `C0` factor
- rotation is reordered into `(x, y, z, w)` usage form

So compared with `DataTable`, `GSplatData` is already closer to "how the renderer thinks".

## 3. PlayCanvas `GSplatResource`

Primary files:

- `reference/engine/src/scene/gsplat/gsplat-resource-base.js`
- `reference/engine/src/scene/gsplat/gsplat-resource.js`
- `reference/engine/src/scene/gsplat/gsplat-streams.js`
- `reference/engine/src/scene/gsplat/gsplat-format.js`

### Core model

`GSplatResource` is where the data stops being a general CPU data model and becomes a GPU stream layout.

It defines resource textures such as:

- `splatColor`
- `transformA`
- `transformB`
- `splatSH_1to3`
- `splatSH_4to7`
- `splatSH_8to11`
- `splatSH_12to15`

These are described by `GSplatFormat`, allocated by `GSplatStreams`, and filled from `GSplatData`.

### What changed from `DataTable`

At this stage the data is no longer represented as independent logical columns. It is repacked into GPU-friendly textures:

- position goes into `transformA`
- rotation is split and half-packed across `transformA` / `transformB`
- scale is exponentiated and half-packed into `transformB`
- color is decoded from SH DC and packed into `splatColor`
- opacity is passed through sigmoid and packed into `splatColor`
- SH coefficients are quantized and packed into integer textures

### Important semantic consequence

`DataTable` keeps authoring / processing values.

`GSplatResource` stores render-ready values.

This means:

- some transforms are already baked
- some precision is intentionally reduced
- some fields are rearranged to match shader fetch patterns instead of human-readable structure

## 4. Supersplat Editor Extensions

Primary file:

- `reference/supersplat/src/splat.ts`

Supersplat adds editor-only structure on top of PlayCanvas runtime splats.

### Added CPU-side columns

Supersplat appends extra per-splat properties:

- `state`
  - `Uint8Array`
  - selected / deleted / locked bitfield
- `transform`
  - `Uint16Array`
  - index into transform palette

These are not part of the core `splat-transform` `DataTable` definition.

### Added GPU-side textures

Supersplat also creates instance/editor textures:

- `splatState`
- `splatTransform`

and a transform palette object that stores actual transform matrices.

So compared with `DataTable`, supersplat introduces a second category of data:

- not source Gaussian attributes
- but live editor/render control state

## 5. Main Structural Differences

## A. Generic table vs render-specific object graph

`DataTable`:

- generic
- reusable
- easy to clone, subset, permute, and transform

PlayCanvas runtime:

- specialized
- built around gsplat loading and rendering
- tied to resource objects, textures, materials, sorters, and mesh instances

## B. Explicit coordinate metadata vs implicit render-space expectations

`DataTable` has:

- `transform: Transform`

This explicitly tracks what coordinate space the raw column values belong to.

PlayCanvas runtime structures do not keep the same abstraction at the data-object layer. Instead:

- CPU data is assumed to already be in the expected semantic space
- conversion happens during processing or resource creation
- final render data is packed for shader consumption

This is one of the biggest conceptual differences.

## C. Raw processing values vs decoded render values

In `DataTable`:

- scale remains log-scale
- opacity remains logit
- SH is still plain coefficient data

In PlayCanvas render structures:

- scale is exponentiated before packing
- opacity is sigmoid-decoded before packing
- DC color is converted to visible color before packing
- SH may be quantized and packed into compact integer textures

## D. Arbitrary columns vs fixed render streams

`DataTable` can hold arbitrary named columns as long as row counts match.

`GSplatResource` uses a fixed or semi-fixed stream contract defined by `GSplatFormat`.

That means:

- `DataTable` is extensible by adding columns
- runtime rendering is extensible by adding texture streams and shader declarations
- these are different extension mechanisms

## E. CPU analytics shape vs GPU fetch shape

`DataTable` is ideal for:

- filtering
- transform conversion
- decimation
- Morton reorder
- clustering
- voxelization

`GSplatResource` is ideal for:

- texture fetches in shader code
- compact packing
- render-time sort and rasterization

So the difference is not just syntax. It is a difference in purpose.

## 6. Field-Level Correspondence

These fields map conceptually one-to-one between `DataTable` and PlayCanvas `GSplatData`:

| Concept | `splat-transform` | PlayCanvas runtime |
|---|---|---|
| Position | `x`,`y`,`z` | `x`,`y`,`z` |
| Rotation | `rot_0..3` | `rot_0..3` |
| Scale | `scale_0..2` | `scale_0..2` |
| DC color | `f_dc_0..2` | `f_dc_0..2` |
| Opacity | `opacity` | `opacity` |
| SH rest | `f_rest_*` | `f_rest_*` |

But they diverge in how they are used:

- `DataTable`
  - processing-time canonical storage
- `GSplatData`
  - runtime semantic storage
- `GSplatResource`
  - GPU-packed render storage

## 7. Extra Data That Exists Only on the Rendering Side

Rendering adds structures that do not exist in `DataTable` itself:

- `centers`
  - CPU-side center cache for sorting
- `aabb`
  - runtime bounds
- `streams`
  - GPU textures grouped by render format
- `mesh`
  - render mesh
- `orderTexture` / `orderBuffer`
  - per-frame sorted splat order
- sorter state
  - camera-dependent ordering
- shader format declarations
  - generated by `GSplatFormat`

Supersplat adds even more rendering/editor-only structures:

- `stateTexture`
- `transformTexture`
- `transformPalette`
- selection / deletion / lock state

## 8. Practical Implications for SplatLib / `ai/superpowers`

If SplatLib wants PlayCanvas-compatible behavior, the safest way to think about the pipeline is:

1. `DataTable`-like structure
   - canonical source-of-truth for processing
2. semantic normalization
   - ensure field names and meanings match PlayCanvas expectations
3. render packing
   - convert into texture-stream layout equivalent to `GSplatResource`
4. editor state layering
   - if needed, add supersplat-style state / transform indirection separately

That separation helps avoid mixing:

- source data semantics
- coordinate-space metadata
- runtime GPU packing
- editor interaction state

## 9. Bottom-Line Difference

The biggest difference is not the field names. Those are mostly compatible.

The biggest difference is the abstraction level:

- `DataTable` is a general, transform-aware, CPU-side processing container
- PlayCanvas rendering structures are staged, specialized, and GPU-oriented

Said another way:

- `DataTable` answers: "what numeric Gaussian attributes do I have?"
- `GSplatResource` answers: "how should shaders fetch and render them?"

