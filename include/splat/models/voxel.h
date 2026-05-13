/**
 * @file voxel.h
 * @brief Voxel grid data structures and conversion utilities.
 *
 * This file defines voxel representation, grid structures and functions
 * for converting Gaussian splats to voxelized volumes.
 */

#pragma once

#include <string>
#include <vector>

namespace splat {

struct VoxelMetadata {
  /** File format version */
  std::string version;

  /** Grid bounds aligned to 4x4x4 block boundaries */
  struct {
    std::vector<double> min;
    std::vector<double> max;
  } gridBounds;

  /** Original Gaussian scene bounds */
  struct {
    std::vector<double> min;
    std::vector<double> max;
  } sceneBounds;

  /** Size of each voxel in world units */
  double voxelResolution{0.0};

  /** Voxels per leaf dimension (always 4) */
  int leafSize{4};

  /** Maximum tree depth */
  int treeDepth{0};

  /** Number of interior nodes */
  int numInteriorNodes{0};

  /** Number of mixed leaf nodes */
  int numMixedLeaves{0};

  /** Total number of Uint32 entries in the nodes array */
  int nodeCount{0};

  /** Total number of Uint32 entries in the leafData array */
  int leafDataCount{0};

  VoxelMetadata(const std::string& json);

  std::string dump() const;
};

}  // namespace splat
