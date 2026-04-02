/***********************************************************************************
 *
 * splat - A C++ library for reading and writing 3D Gaussian Splatting (splat) files.
 *
 * This file is part of splat.
 *
 * splat is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 ***********************************************************************************/

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
