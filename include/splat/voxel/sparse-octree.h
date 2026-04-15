/**
 * @file splat/voxel/sparse-octree.h
 * @brief Sparse voxel octree structures.
 *
 * References:
 * - [Octree](https://en.wikipedia.org/wiki/Octree)
 */

#pragma once

#include <splat/maths/maths.h>

#include <Eigen/Core>
#include <cstdint>
#include <vector>

/**
 * @file sparse-octree.h
 * @brief Sparse voxel octree implementation for efficient storage of voxel data
 */

namespace splat {

/**
 * @brief Accumulator for streaming voxelization results
 *
 * Stores blocks using Morton codes for efficient octree construction.
 */
class BlockAccumulator {
 public:
  /**
   * @brief Add a non-empty block to the accumulator
   * @param morton Morton code encoding block position
   * @param lo Lower 32 bits of voxel mask
   * @param hi Upper 32 bits of voxel mask
   */
  void addBlock(uint32_t morton, uint32_t lo, uint32_t hi);

  /**
   * @brief Structure holding mixed block data
   */
  struct MixedBlocks {
    std::vector<uint32_t> morton;  ///< Morton codes for mixed blocks
    std::vector<uint32_t> masks;   ///< Interleaved masks: [lo0, hi0, lo1, hi1, ...]
  };

  /**
   * @brief Get all mixed blocks
   * @return MixedBlocks with morton codes and interleaved masks
   */
  MixedBlocks getMixedBlocks() const;

  /**
   * @brief Get all solid blocks
   * @return Vector of Morton codes for solid blocks
   */
  const std::vector<uint32_t>& getSolidBlocks() const;

  /**
   * @brief Get total number of blocks stored
   * @return Count of mixed + solid blocks
   */
  size_t count() const;

  /**
   * @brief Get number of mixed blocks
   * @return Count of mixed blocks
   */
  size_t mixedCount() const;

  /**
   * @brief Get number of solid blocks
   * @return Count of solid blocks
   */
  size_t solidCount() const;

  /**
   * @brief Clear all accumulated blocks
   */
  void clear();

 private:
  std::vector<uint32_t> mixedMorton_;  ///< Morton codes for mixed blocks
  std::vector<uint32_t> mixedMasks_;   ///< Interleaved voxel masks: [lo0, hi0, ...]
  std::vector<uint32_t> solidMorton_;  ///< Morton codes for solid blocks
};

// ============================================================================
// Sparse Octree Types
// ============================================================================

/**
 * @brief Bounds specification with min/max Vec3
 */
struct Bounds {
  Eigen::Vector3f min;  ///< Minimum corner of the bounding box
  Eigen::Vector3f max;  ///< Maximum corner of the bounding box
};

/**
 * @brief Sparse voxel octree using Laine-Karras node format
 *
 * This structure stores a sparse octree in a compact array-based format
 * suitable for GPU consumption and efficient memory usage.
 */
struct SparseOctree {
  Bounds gridBounds;               ///< Grid bounds aligned to 4x4x4 block boundaries
  Bounds sceneBounds;              ///< Original Gaussian scene bounds
  float voxelResolution;           ///< Size of each voxel in world units
  int leafSize;                    ///< Voxels per leaf dimension (always 4)
  int treeDepth;                   ///< Maximum tree depth
  int numInteriorNodes;            ///< Number of interior nodes
  int numMixedLeaves;              ///< Number of mixed leaf nodes
  std::vector<uint32_t> nodes;     ///< All nodes in Laine-Karras format
  std::vector<uint32_t> leafData;  ///< Voxel masks for mixed leaves: pairs of u32 (lo, hi)
};

// ============================================================================
// Octree Construction
// ============================================================================

/**
 * @brief Build a sparse octree from accumulated voxelization blocks
 *
 * Uses Structure-of-Arrays (SoA) representation and linear scans on sorted
 * Morton codes instead of Maps and per-node objects for performance.
 *
 * @param accumulator BlockAccumulator containing voxelized blocks
 * @param gridBounds Grid bounds aligned to block boundaries
 * @param sceneBounds Original scene bounds
 * @param voxelResolution Size of each voxel in world units
 * @return SparseOctree structure in Laine-Karras format
 */
SparseOctree buildSparseOctree(const BlockAccumulator& accumulator, const Bounds& gridBounds, const Bounds& sceneBounds,
                               float voxelResolution);

/**
 * @brief Align bounds to 4x4x4 block boundaries
 * @param minX Scene minimum X
 * @param minY Scene minimum Y
 * @param minZ Scene minimum Z
 * @param maxX Scene maximum X
 * @param maxY Scene maximum Y
 * @param maxZ Scene maximum Z
 * @param voxelResolution Size of each voxel
 * @return Aligned bounds
 */
Bounds alignGridBounds(float minX, float minY, float minZ, float maxX, float maxY, float maxZ, float voxelResolution);

}  // namespace splat
