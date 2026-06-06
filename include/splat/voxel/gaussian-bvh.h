/**
 * @file splat/voxel/gaussian-bvh.h
 * @brief BVH over Gaussians for acceleration.
 *
 */

#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <vector>

/**
 * @file gaussian-bvh.h
 * @brief Bounding Volume Hierarchy for efficient spatial queries on Gaussian AABBs
 */

namespace splat {

class SplatCloud;

/**
 * @brief Axis-aligned bounding box for BVH nodes
 */
struct BVHBounds {
  float minX, minY, minZ;
  float maxX, maxY, maxZ;
};

/**
 * @brief BVH (Bounding Volume Hierarchy) for efficient spatial queries on Gaussian AABBs
 *
 * Unlike the centroid-based BTree, this BVH stores the full AABB for each node,
 * computed from position +/- extent for all Gaussians in the subtree.
 */
class GaussianBVH {
 public:
  /**
   * @brief Construct a BVH from Gaussian data
   * @param dataTable SplatCloud containing position (x, y, z) columns
   * @param extents SplatCloud containing extent (extent_x, extent_y, extent_z) columns
   */
  GaussianBVH(const SplatCloud& dataTable, const SplatCloud& extents);

  /**
   * @brief Query all Gaussian indices whose AABBs overlap the given box
   * @param boxMin Minimum corner of query box
   * @param boxMax Maximum corner of query box
   * @return Vector of Gaussian indices that overlap the box
   */
  std::vector<uint32_t> queryOverlapping(const Eigen::Vector3f& boxMin, const Eigen::Vector3f& boxMax) const;

  /**
   * @brief Query all Gaussian indices whose AABBs overlap the given box (raw coordinates)
   * @param minX Minimum X of query box
   * @param minY Minimum Y of query box
   * @param minZ Minimum Z of query box
   * @param maxX Maximum X of query box
   * @param maxY Maximum Y of query box
   * @param maxZ Maximum Z of query box
   * @return Vector of Gaussian indices that overlap the box
   */
  std::vector<uint32_t> queryOverlappingRaw(float minX, float minY, float minZ, float maxX, float maxY,
                                            float maxZ) const;

  /**
   * @brief Get the total number of Gaussians in the BVH
   * @return Total count of Gaussians
   */
  size_t count() const { return count_; }

  /**
   * @brief Get the bounds of the entire scene
   * @return Scene bounding box
   */
  const BVHBounds& sceneBounds() const { return rootBounds_; }

 private:
  // BVH node structure
  struct BVHNode {
    BVHBounds bounds;
    uint32_t count;                 // Number of Gaussians in this subtree
    uint32_t leftOffset;            // Offset to left child (0 if leaf)
    uint32_t rightOffset;           // Offset to right child (0 if leaf)
    std::vector<uint32_t> indices;  // Only for leaf nodes
  };

  // Build a BVH node recursively
  size_t buildNode(std::vector<uint32_t>& indices, size_t offset, size_t count);

  // Query helper
  void queryNode(const Eigen::Vector3f& min, const Eigen::Vector3f& max, size_t nodeIdx,
                 std::vector<uint32_t>& result) const;

  // Check if two AABBs overlap
  bool boundsOverlap(const BVHBounds& a, float bMinX, float bMinY, float bMinZ, float bMaxX, float bMaxY,
                     float bMaxZ) const;

  std::vector<BVHNode> nodes_;  // BVH nodes stored in array
  BVHBounds rootBounds_;        // Root node bounds
  size_t count_;                // Total Gaussian count

  // Column data references (non-owning)
  const std::vector<float>* x_{nullptr};
  const std::vector<float>* y_{nullptr};
  const std::vector<float>* z_{nullptr};
  const std::vector<float>* extentX_{nullptr};
  const std::vector<float>* extentY_{nullptr};
  const std::vector<float>* extentZ_{nullptr};

  // Maximum leaf size
  static constexpr size_t MAX_LEAF_SIZE = 64;
};

}  // namespace splat
