/**
 * @file splat/voxel/nav-simplify.h
 * @brief Navigation mesh simplification helpers.
 *
 * References:
 * - [meshoptimizer](https://github.com/zeux/meshoptimizer)
 */

#pragma once

#include <splat/voxel/sparse-octree.h>

#include <memory>

/**
 * @file nav-simplify.h
 * @brief Navigation simplification for capsule-based collision detection
 */

namespace splat {

/**
 * @brief Seed position for capsule navigation simplification
 */
struct NavSeed {
  float x, y, z;  ///< World space coordinates
};

/**
 * @brief Result of capsule navigation simplification
 */
struct NavSimplifyResult {
  std::unique_ptr<BlockAccumulator> accumulator;  ///< Simplified voxel data
  Bounds gridBounds;                              ///< Cropped grid bounds
};

/**
 * @brief Simplify voxel collision data for upright capsule navigation
 *
 * Uses bitfield storage (1 bit per voxel) to reduce memory by 8x compared
 * to byte-per-voxel. Two buffers are ping-ponged through the dilation,
 * BFS, inversion, and erosion phases.
 *
 * Algorithm:
 * 1. Build dense bitfield grid from the accumulator.
 * 2. Dilate solid by the capsule shape (Minkowski sum) to get clearance grid.
 * 3. BFS flood fill from seed to find reachable positions.
 * 4. Invert: non-reachable cells become solid.
 * 5. Erode solid by capsule shape to return surfaces to original positions.
 * 6. Crop to bounding box of navigable cells.
 *
 * @param accumulator BlockAccumulator with filtered voxelization results
 * @param gridBounds Grid bounds aligned to block boundaries
 * @param voxelResolution Size of each voxel in world units
 * @param capsuleHeight Total capsule height in world units
 * @param capsuleRadius Capsule radius in world units
 * @param seed Seed position in world space
 * @return Simplified accumulator and cropped grid bounds
 */
NavSimplifyResult simplifyForCapsule(const BlockAccumulator& accumulator, const Bounds& gridBounds,
                                     float voxelResolution, float capsuleHeight, float capsuleRadius,
                                     const NavSeed& seed);

}  // namespace splat
