/***********************************************************************************
 *
 * splat - A C++ library for reading and writing 3D Gaussian Splatting (splat) files.
 *
 * This library provides functionality to convert, manipulate, and process
 * 3D Gaussian splatting data formats used in real-time neural rendering.
 *
 * This file is part of splat.
 *
 * splat is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * splat is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * For more information, visit the project's homepage or contact the author.
 *
 ***********************************************************************************/

#pragma once

#include <cstdint>
#include <memory>

#include "splat/voxel/sparse-octree.h"

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
