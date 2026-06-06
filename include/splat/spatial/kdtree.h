/**
 * @file splat/spatial/kdtree.h
 * @brief k-d tree on `SplatCloud` for spatial queries.
 *
 * References:
 * - [k-d tree](https://en.wikipedia.org/wiki/K-d_tree)
 */

#pragma once

#include <absl/types/span.h>

#include <functional>
#include <memory>
#include <vector>

namespace splat {

class SplatCloud;

/**
 * @brief K-dimensional tree for efficient spatial queries on 3D Gaussian splatting data.
 *
 * This class implements a k-d tree data structure optimized for finding nearest neighbors
 * in 3D space. It is specifically designed for use with 3D Gaussian splatting data where
 * efficient spatial queries are required for real-time neural rendering applications.
 */
class KdTree {
 public:
  /**
   * @brief Node structure representing a single node in the k-d tree.
   *
   * Each node contains a reference to a data point and pointers to its left and right children.
   * The tree is built recursively by partitioning the data along alternating dimensions.
   */
  struct KdTreeNode {
    size_t index;                       ///< Index of the data point in the original dataset
    size_t count;                       ///< Number of points in the subtree rooted at this node
    std::unique_ptr<KdTreeNode> left;   ///< Left child node (points with smaller values in the splitting dimension)
    std::unique_ptr<KdTreeNode> right;  ///< Right child node (points with larger values in the splitting dimension)

    /**
     * @brief Construct a new KdTreeNode.
     *
     * @param index Index of the data point in the original dataset
     * @param count Number of points in the subtree rooted at this node
     * @param left Left child node
     * @param right Right child node
     */
    KdTreeNode(size_t index, size_t count, std::unique_ptr<KdTreeNode> left, std::unique_ptr<KdTreeNode> right)
        : index(index), count(count), left(std::move(left)), right(std::move(right)) {}
  };

 public:
  /**
   * @brief Indices for the tuple returned by findNearest method.
   */
  enum {
    index,                   ///< Index of the nearest neighbor in the dataset
    distanceSqr,             ///< Squared distance to the nearest neighbor
    count,                   ///< Number of nodes visited during the search
    findNearestMaxIndex = 3  ///< Maximum valid index for the tuple
  };

  /**
   * @brief Construct a new KdTree from a SplatCloud.
   *
   * The k-d tree is built from the centroids stored in the provided SplatCloud.
   * The tree construction uses the median of points along alternating dimensions
   * to ensure a balanced tree structure.
   *
   * @param table Pointer to the SplatCloud containing the data points (centroids)
   *              The table must not be null and should contain valid data
   */
  KdTree(SplatCloud* table);

  /**
   * @brief Find the nearest neighbor to a given point.
   *
   * This method performs an efficient nearest neighbor search in the k-d tree.
   * It uses a recursive algorithm that explores the tree while maintaining a
   * bounding sphere to prune unnecessary branches.
   *
   * @param point The query point for which to find the nearest neighbor
   *              Must have the same dimensionality as the data points in the tree
   * @param filterFunc Optional filter function to exclude certain points from consideration
   *                   Should return true for points that are allowed to be considered
   * @return std::tuple<int, float, size_t> A tuple containing:
   *         - The index of the nearest neighbor (or -1 if not found)
   *         - The squared distance to the nearest neighbor
   *         - The number of nodes visited during the search
   */
  std::tuple<int, float, size_t> findNearest(const std::vector<float>& point,
                                             std::function<bool(size_t)> filterFunc = nullptr);

  /**
   * @brief Find up to k nearest neighbors (squared Euclidean in column space).
   * @param point Query coordinates (length must match table column count)
   * @param k Maximum neighbors to return (may be fewer if filtered or data is small)
   * @param filterFunc Optional per-index filter (true = candidate allowed)
   */
  std::vector<size_t> findKNearest(const std::vector<float>& point, size_t k,
                                   std::function<bool(size_t)> filterFunc = nullptr);

  /**
   * @brief Find up to k nearest neighbors into a caller-owned output vector.
   * @param point Query coordinates. Must contain one value per tree column.
   * @param k Maximum neighbors to return.
   * @param out Output indices, replaced on each call.
   * @param filterFunc Optional per-index filter (true = candidate allowed)
   */
  void findKNearest(const float* point, size_t k, std::vector<size_t>& out,
                    std::function<bool(size_t)> filterFunc = nullptr);

 private:
  SplatCloud* centroids;             ///< Pointer to the SplatCloud containing the data points
  std::unique_ptr<KdTreeNode> root;  ///< Root node of the k-d tree
  std::vector<const std::vector<float>*> columnData;
  std::vector<float> heapDistances;
  std::vector<size_t> heapIndices;

  /**
   * @brief Recursively build the k-d tree from a set of indices.
   *
   * This private method constructs the k-d tree by recursively partitioning
   * the data points. At each level, it selects the median point along the
   * current splitting dimension and recursively builds the left and right subtrees.
   *
   * @param indices Span of indices to the data points to be included in this subtree
   * @param depth Current depth in the tree (used to determine the splitting dimension)
   * @return std::unique_ptr<KdTreeNode> Root node of the constructed subtree
   */
  std::unique_ptr<KdTreeNode> build(absl::Span<size_t> indices, size_t depth);
};

}  // namespace splat
