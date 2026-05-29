# Decimate Algorithm: Follow Upstream splat-transform v2.1.1 → v2.4.0 (P0–P1)

**Date:** 2026-05-29

## Goal

Port the area-weighted, mass-conserving pairwise merge math from splat-transform upstream (#241) and apply the structural Float64→Float32 + scratch-buffer improvements (#244) to the C++ decimate implementation. Fix color drift and over-saturation at aggressive reduction ratios, reduce memory usage ~50% for the per-splat cache, and eliminate per-call heap allocations in the merge loop.

## Background

The C++ SplatLib decimate code at `src/op/decimate.cpp` and `include/splat/maths/gaussian-decimate-math.h` was ported from splat-transform v2.1.1 (pinned at `bebac61`). Since then, upstream v2.2.0 through v2.4.0 made two important changes to the decimate algorithm:

**#241 — Fix decimate color drift.** The merge formula switched from NanoGS (volume·α weighting, Porter-Duff opacity, mass-weighted color) to the spark model (ellipsoid-area·α weighting, mass-conserving opacity clamped at 1, area·α weighted normalized color). The commit message states: *"color drift / over-saturation is essentially gone even at aggressive reductions."*

**#244 — GPU-accelerate KNN and edge cost.** While the GPU compute-shader path itself is WebGPU-specific, the structural improvements apply universally:
- SplatCache fields changed from Float64 to Float32 (single-precision sufficient for the MC-sample heuristic; halves memory)
- `Rt` (transpose) removed from cache — derivable from `R` via index swap
- Per-call `std::array`/`Float64Array` allocations replaced with a `MergeScratch` struct allocated once per `simplifyGaussians` invocation
- Pre-merge opacity pruning step removed

The upstream analysis is recorded in `docs/superpowers/reference/upstream-hashes.md`.

## Design Decision

Apply all P0–P1 algorithmic and structural fixes to the existing two-file implementation. Do **not** add GPU acceleration, new rendering features, or new public API. The changes are:

1. **Add `ellipsoid_area()`** to the math header — Knud Thomsen p=1.6075 approximation for ellipsoid surface area, used as the per-splat weight in both merge cost and moment matching.

2. **Add caller-provided-buffer overload of `eigen_symmetric_3x3()`** — the existing function allocates internal `A[9]` and `V[9]` arrays and returns by value. The new overload accepts caller-owned scratch so the merge loop can reuse buffers across thousands of calls.

3. **Switch `SplatCache` to Float32, remove `Rt`, use area·α mass** — all six storage arrays (`R`, `v`, `invdiag`, `logdet`, `sigma`, `mass`) change from `std::vector<double>` to `std::vector<float>`. `Rt` is dropped. Mass changes from `(2π)^1.5·α·sx·sy·sz` to `α·ellipsoid_area(sx,sy,sz)`.

4. **Add `MergeScratch` struct** — nine `std::array<double,9>` fields pre-allocated once per `simplifyGaussians()` call and passed by reference into `compute_edge_cost()` and `moment_match()`. Eliminates ~9M per-call heap allocations in a large decimate.

5. **Rewrite `compute_edge_cost()`** — use `cache.R` directly (row-major, `x = mu + R·diag(std)·z`) instead of `cache.Rt`. Accept `MergeScratch&` for the sigm buffer. Convert cache float values to double at point of use for MC sampling precision.

6. **Rewrite `moment_match()`** — replace all four formulas:

   | Aspect | Old (NanoGS) | New (spark) |
   |--------|-------------|-------------|
   | Weight | (2π)^1.5·sx·sy·sz·α | ellipsoidArea(sx,sy,sz)·α |
   | Mean | mass-weighted by wi/W | area·α-weighted by pi = wi/W |
   | Opacity | α_i + α_j − α_i·α_j | min(1, Σα·area / area_merged) |
   | Color | mass-weighted / W | area·α weighted pi·v_i + pj·v_j |

7. **Remove pre-merge opacity pruning** — upstream dropped it (v2.2.0+). The area·α merge math handles low-opacity splats through natural weight attenuation.

## Architecture

Changes are localized to two files:

```text
include/splat/maths/gaussian-decimate-math.h   ← Add ellipsoid_area(), eigen_symmetric_3x3(Ain, A, V)
src/op/decimate.cpp                              ← SplatCache f32, MergeScratch, rewrite merge/cost formulas, remove opacity pruning
```

No new files. No changes to the public API (`simplifyGaussians`, `sortByVisibility`). No new dependencies.

## Components

### `ellipsoid_area()` — new function in math header

Knud Thomsen p=1.6075 approximation. Three semi-axes in, scalar surface area out. Used in `build_per_splat_cache` (for mass), `moment_match` (for weights and mass-conserving opacity).

### `eigen_symmetric_3x3(Ain, A, V)` — new overload in math header

Same Jacobi algorithm. `A` is overwritten with eigenvalues on its diagonal (positions 0,4,8). `V` is overwritten with eigenvectors as columns. Caller owns both buffers. The existing single-argument overload is preserved for any external callers.

### `SplatCache` — struct rewrite

`std::vector<float>` throughout. `Rt` removed — call sites use `R` with the convention `R[row + col]` for sampling, matching the GPU kernel form `x = mu + R·diag(std)·z`. Mass computed as `α·ellipsoid_area(sx,sy,sz) + 1e-12f`.

### `MergeScratch` — new struct

Nine `std::array<double,9>` fields: `sigm`, `sigI`, `sigJ`, `rI`, `rJ`, `sig`, `rM`, `eigA`, `eigV`. Instantiated once at the top of `simplifyGaussians()`, passed by reference to `compute_edge_cost()` and `moment_match()`. Stack-allocated (no heap).

### `compute_edge_cost()` — function rewrite

Same KL-divergence cost logic, but: uses `cache.R` (not `cache.Rt`) for MC sampling; uses `scratch.sigm` for the merged covariance buffer; converts float cache values to double at point of use; weights are now area·α (via the changed `cache.mass`).

### `moment_match()` — function rewrite

All four merge formulas changed (table above). Eigendecomposition uses the caller-buffer overload with `scratch.eigA`/`scratch.eigV`. Eigenvalue sorting uses a hand-unrolled branch tree instead of `std::sort`. Scale output is `log(sqrt(eigenvalue))` (equivalent to old `0.5·log(eigenvalue)`, more direct).

### `simplifyGaussians()` — opacity pruning removal

The pre-merge opacity pruning block (~20 lines) is deleted. A `MergeScratch` is instantiated after the MC samples. Call sites for `compute_edge_cost` and `moment_match` are updated to pass the scratch reference.

## Breaking Changes

None. Public API unchanged. Output quality improves (no color drift, valid opacity range). Memory usage ~50% lower for SplatCache arrays.
