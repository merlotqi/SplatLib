/**
 * @file splat/voxel/voxel-filter.h
 * @brief Filter / fill voxel block masks.
 *
 */

#pragma once

#include <splat/voxel/sparse-octree.h>

namespace splat {

/**
 * @brief Remove isolated voxels and fill isolated empty voxels within mixed blocks.
 *
 * For each mixed block, computes 6 per-direction occupancy masks (in-block via
 * bit shifts + cross-block via adjacent block lookups), then:
 *   - Remove: keeps only voxels with at least one occupied 6-connected neighbor
 *   - Fill: fills empty voxels where all 6 neighbors are occupied
 *
 * Cross-block lookups always read the original (pre-filter) mixed masks; solid
 * blocks are preserved and re-added unchanged. Semantics match
 * splat-transform/lib/voxel/voxel-filter.ts.
 *
 * @param accumulator BlockAccumulator with voxelization results
 * @return New BlockAccumulator with filtered/filled data
 */
BlockAccumulator filterAndFillBlocks(const BlockAccumulator& accumulator);

}  // namespace splat
