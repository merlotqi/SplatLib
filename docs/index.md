---
title: Splat File Formats
---

PlayCanvas supports two formats for 3D Gaussian Splat data:

## [PLY Format](./ply.md) - Source & Interchange

The industry standard for Gaussian splat data. Uncompressed, full precision, and universally compatible.

- **Use for**: Training, editing, archival storage
- **File size**: Large (anything up to several GB)
- **Quality**: Lossless

## [SOG Format](./sog.md) - Runtime & Delivery

Compressed format optimized for web delivery. 15-20× smaller than PLY with lossy compression.

- **Use for**: Web apps, real-time rendering, CDN delivery
- **File size**: Small (compressed)
- **Quality**: Visually optimized

## Quick Comparison

| | PLY | SOG |
|---|---|---|
| **Size** | Large | Small (15-20× compression) |
| **Quality** | Lossless | Lossy |
| **Use** | Source/editing | Runtime/delivery |
| **Speed** | Slow loading | Fast loading |

## Workflow

1. Train and edit with **PLY**
2. Deploy SOG files for optimal performance

## [PlayCanvas 3DGS Code Map](./superpowers/reference/playcanvas-3dgs-code-map.md) - Engine & Supersplat Internals

Reference guide to the PlayCanvas 3D Gaussian Splatting implementation imported into the `ai/superpowers` worktree.

- **Use for**: Code navigation, render pipeline tracing, shader lookup
- **Covers**: `reference/engine`, `reference/supersplat`, data structures, rendering flow, shaders

## [Splat-Transform DataTable vs PlayCanvas Rendering Structures](./superpowers/reference/splat-transform-data-table-vs-rendering.md) - Data Model Comparison

Comparison between `reference/splat-transform`'s `DataTable` model and the PlayCanvas runtime/editor rendering structures.

- **Use for**: Data model alignment, TS-to-render structure comparison, integration design
- **Covers**: `DataTable`, `GSplatData`, `GSplatResource`, supersplat editor state

## [Visualization Module Structure](./superpowers/reference/visualization-module-structure.md) - Native Visualization Branch Analysis

Structure analysis of the historical visualization implementation found in `feature/visualization`, including both the reusable VTK-based module and the standalone viewer app.

- **Use for**: Native visualization architecture review, VTK/OpenGL render path tracing, historical module understanding
- **Covers**: `include/splat/visualization`, `src/visualization`, `viewer/`, event system, rendering flow
