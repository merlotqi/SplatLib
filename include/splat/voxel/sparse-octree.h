/**
 * @file splat/voxel/sparse-octree.h
 * @brief Sparse voxel octree structures.
 *
 * References:
 * - [Octree](https://en.wikipedia.org/wiki/Octree)
 */
 
#pragma once

#include <Eigen/Core>

#include <array>
#include <cstdint>
#include <vector>

/**
 * @file sparse-octree.h
 * @brief Sparse voxel octree implementation for efficient storage of voxel data
 */

namespace splat {

// ============================================================================
// Constants
// ============================================================================

/// All 64 bits set (as unsigned 32-bit)
inline constexpr uint32_t SOLID_MASK = 0xFFFFFFFFu;

/// Solid leaf node marker: childMask = 0xFF, baseOffset = 0
inline constexpr uint32_t SOLID_LEAF_MARKER = 0xFF000000u;

// ============================================================================
// Morton Code Functions
// ============================================================================

/**
 * @brief Encode block coordinates to Morton code (17 bits per axis = 51 bits total)
 * @param x Block X coordinate
 * @param y Block Y coordinate
 * @param z Block Z coordinate
 * @return Morton code with interleaved bits
 */
inline uint32_t xyzToMorton(uint32_t x, uint32_t y, uint32_t z) {
  uint32_t result = 0;
  uint32_t shift = 1;
  for (int i = 0; i < 17; i++) {
    if (x & 1) result += shift;
    if (y & 1) result += shift * 2;
    if (z & 1) result += shift * 4;
    x >>= 1;
    y >>= 1;
    z >>= 1;
    shift *= 8;
  }
  return result;
}

/**
 * @brief Decode Morton code to block coordinates
 * @param m Morton code
 * @return Tuple of [x, y, z] block coordinates
 */
inline std::array<uint32_t, 3> mortonToXYZ(uint32_t m) {
  uint32_t x = 0, y = 0, z = 0;
  uint32_t bit = 1;
  while (m > 0) {
    uint32_t triplet = m % 8;
    if (triplet & 1) x |= bit;
    if (triplet & 2) y |= bit;
    if (triplet & 4) z |= bit;
    bit <<= 1;
    m /= 8;
  }
  return {x, y, z};
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Count the number of set bits in a 32-bit integer
 * @param n 32-bit integer
 * @return Number of bits set to 1
 */
inline int popcount(uint32_t n) {
  n -= ((n >> 1) & 0x55555555u);
  n = (n & 0x33333333u) + ((n >> 2) & 0x33333333u);
  return static_cast<int>(((n + (n >> 4)) & 0x0F0F0F0Fu) * 0x01010101u >> 24);
}

/**
 * @brief Check if a voxel mask represents a solid block (all 64 bits set)
 * @param lo Lower 32 bits of mask
 * @param hi Upper 32 bits of mask
 * @return True if all 64 voxels are solid
 */
inline bool isSolid(uint32_t lo, uint32_t hi) {
  return lo == SOLID_MASK && hi == SOLID_MASK;
}

/**
 * @brief Check if a voxel mask represents an empty block (no bits set)
 * @param lo Lower 32 bits of mask
 * @param hi Upper 32 bits of mask
 * @return True if all 64 voxels are empty
 */
inline bool isEmpty(uint32_t lo, uint32_t hi) { return lo == 0 && hi == 0; }

/**
 * @brief Get the offset to a child node given a parent's child mask and octant
 * @param mask 8-bit child mask from parent node
 * @param octant Octant index (0-7)
 * @return Offset from base child pointer
 */
inline int getChildOffset(uint8_t mask, int octant) {
  uint8_t prefix = static_cast<uint8_t>((1u << octant) - 1);
  return popcount(mask & prefix);
}

// ============================================================================
// Block Accumulator
// ============================================================================

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
  Bounds gridBounds;                  ///< Grid bounds aligned to 4x4x4 block boundaries
  Bounds sceneBounds;                 ///< Original Gaussian scene bounds
  float voxelResolution;              ///< Size of each voxel in world units
  int leafSize;                       ///< Voxels per leaf dimension (always 4)
  int treeDepth;                      ///< Maximum tree depth
  int numInteriorNodes;               ///< Number of interior nodes
  int numMixedLeaves;                 ///< Number of mixed leaf nodes
  std::vector<uint32_t> nodes;        ///< All nodes in Laine-Karras format
  std::vector<uint32_t> leafData;     ///< Voxel masks for mixed leaves: pairs of u32 (lo, hi)
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
SparseOctree buildSparseOctree(const BlockAccumulator& accumulator, const Bounds& gridBounds,
                               const Bounds& sceneBounds, float voxelResolution);

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
Bounds alignGridBounds(float minX, float minY, float minZ, float maxX, float maxY, float maxZ,
                       float voxelResolution);

}  // namespace splat
