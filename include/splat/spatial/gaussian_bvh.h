/**
 * @file gaussian_bvh.h
 * @brief Bounding Volume Hierarchy spatial acceleration structure for Gaussian splats.
 *
 * This file provides BVH construction, traversal and query operations
 * for efficient spatial indexing of 3D Gaussian point clouds.
 */

#pragma once

#include <absl/types/span.h>

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <vector>

namespace splat {

class SplatCloud;

/**
 * @class GaussianBVH
 * @brief Bounding Volume Hierarchy spatial index for Gaussian splat datasets.
 *
 * Provides efficient spatial query operations over 3D Gaussian point clouds
 * using an axis-aligned bounding box hierarchy tree structure.
 */
class GaussianBVH {
 public:
  /**
   * @struct BVHBounds
   * @brief Axis-aligned bounding box representation.
   */
  struct BVHBounds {
    Eigen::Vector3f min;  ///< Minimum corner coordinate
    Eigen::Vector3f max;  ///< Maximum corner coordinate
  };

  /**
   * @struct BVHNode
   * @brief Hierarchy tree node structure.
   *
   * Internal nodes contain child pointers, leaf nodes contain indices
   * referencing entries in the source SplatCloud.
   */
  struct BVHNode {
    size_t count;                    ///< Number of elements in this subtree
    BVHBounds bounds;                ///< Bounding box covering all elements
    std::vector<uint32_t> indices;   ///< Indices to SplatCloud rows (leaf nodes only)
    std::unique_ptr<BVHNode> left;   ///< Left child subtree
    std::unique_ptr<BVHNode> right;  ///< Right child subtree

    BVHNode() = default;
    BVHNode(size_t count, const BVHBounds& bounds, std::vector<uint32_t> indices)
        : count(count), bounds(bounds), indices(std::move(indices)), left(nullptr), right(nullptr) {}
  };

 public:
  /**
   * @brief Construct BVH acceleration structure from Gaussian dataset.
   * @param dataTable Source Gaussian point cloud data
   * @param extents Precomputed Gaussian AABB half-extents from computeGaussianExtents()
   */
  GaussianBVH(const SplatCloud* dataTable, const SplatCloud* extents);

  /**
   * @brief Query all Gaussian splats overlapping the given bounding box.
   * @param boxMin Minimum corner of query volume
   * @param boxMax Maximum corner of query volume
   * @return Vector of indices referencing overlapping entries in the source SplatCloud
   */
  std::vector<uint32_t> queryOverlapping(const Eigen::Vector3f& boxMin, const Eigen::Vector3f& boxMax);

  /** @return Total number of Gaussian splats in the BVH */
  size_t count() const { return root_ ? root_->count : 0; }

  /** @return Bounding box covering the entire scene */
  BVHBounds sceneBounds() const { return root_ ? root_->bounds : BVHBounds(); }

  /** @return Root node of the BVH tree */
  BVHNode* root() const { return root_.get(); }

 private:
  BVHBounds computeBound(absl::Span<uint32_t> indices);
  std::unique_ptr<BVHNode> buildNode(absl::Span<uint32_t> indices);
  void queryNode(const BVHNode* node, float minX, float minY, float minZ, float maxX, float maxY, float maxZ,
                 std::vector<uint32_t>& result);

 private:
  std::unique_ptr<BVHNode> root_;
  absl::Span<const float> x_;
  absl::Span<const float> y_;
  absl::Span<const float> z_;

  absl::Span<const float> extentX_;
  absl::Span<const float> extentY_;
  absl::Span<const float> extentZ_;

  static constexpr size_t MAX_LEAF_SIZE = 64u;
};

}  // namespace splat
