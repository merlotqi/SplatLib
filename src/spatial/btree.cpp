#include <splat/models/data-table.h>
#include <splat/spatial/btree.h>

#include <numeric>

namespace splat {

/**
 * @brief Partitions the index array (idx) around the k-th smallest element based on the values in 'data'.
 * Implements the Quickselect algorithm with Median-of-Three pivot selection.
 *
 * @param data The data array whose values are used for comparison.
 * @param idx The index array to be partitioned (in-place modification).
 * @param k The index (0-based) of the element to select (e.g., k=indices.size()/2 for median).
 * @return The value of the k-th element in the original index array (before partitioning).
 */
uint32_t quickselect(absl::Span<const float> data, absl::Span<uint32_t> idx, size_t k_in) {
  if (idx.empty()) return 0;

  int n = static_cast<int>(idx.size());
  int l = 0;
  int r = n - 1;
  int k = static_cast<int>(k_in);

  auto valAt = [&](int p) {
    auto ix = idx[p];
    auto v = data[ix];
    return v;
  };
  auto swap = [&](int i, int j) {
    uint32_t t = idx[i];
    idx[i] = idx[j];
    idx[j] = t;
  };

  while (true) {
    if (r <= l + 1) {
      if (r == l + 1 && valAt(r) < valAt(l)) swap(l, r);
      return idx[k];
    }

    // Median-of-three
    int mid = (l + r) / 2;
    swap(mid, l + 1);
    if (valAt(l) > valAt(r)) swap(l, r);
    if (valAt(l + 1) > valAt(r)) swap(l + 1, r);
    if (valAt(l) > valAt(l + 1)) swap(l, l + 1);

    int i = l + 1;
    int j = r;
    float pivotVal = valAt(l + 1);
    uint32_t pivotIdx = idx[l + 1];

    while (true) {
      do {
        i++;
      } while (i <= r && valAt(i) < pivotVal);
      do {
        j--;
      } while (j >= l && valAt(j) > pivotVal);

      if (j < i) break;
      swap(i, j);
    }

    idx[l + 1] = idx[j];
    idx[j] = pivotIdx;
    if (j >= k) r = j - 1;
    if (j <= k) l = i;
  }
}

BTree::AABB::AABB(const std::vector<float>& min, const std::vector<float>& max) : min(min), max(max) {}

/**
 * @brief Calculates the index (0-based) of the largest axis (dimension) of the AABB.
 * @return The index of the largest axis, or -1 if min/max vectors are empty.
 */
int BTree::AABB::largestAxis() const {
  const size_t length = min.size();
  if (length == 0) return -1;
  auto maxRange = -std::numeric_limits<float>::infinity();
  int result = -1;
  for (size_t i = 0; i < length; i++) {
    const float e = max[i] - min[i];
    if (e > maxRange) {
      maxRange = e;
      result = static_cast<int>(i);
    }
  }
  return result;
}

/**
 * @brief Calculates the length of the AABB along its largest dimension.
 * @return The largest dimension size, or 0 if empty.
 */
float BTree::AABB::largestDim() const {
  const auto a = largestAxis();
  return max[a] - min[a];
}

/**
 * @brief Computes the AABB that tightly encloses the centroids specified by the indices.
 * Resizes min/max vectors if necessary.
 * @param centroids The DataTable containing the centroid coordinate columns.
 * @param indices The indices of the rows to include in the AABB calculation.
 * @return A reference to the updated Aabb object.
 */
BTree::AABB& BTree::AABB::fromCentroids(const DataTable* centroids, absl::Span<const uint32_t> indices) {
  const size_t numColumns = centroids->getNumColumns();
  min.assign(numColumns, std::numeric_limits<float>::infinity());
  max.assign(numColumns, -std::numeric_limits<float>::infinity());

  for (size_t j = 0; j < numColumns; j++) {
    auto&& data = centroids->getColumn(j).asSpan<float>();
    float m = std::numeric_limits<float>::infinity();
    float M = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < indices.size(); ++i) {
      auto&& v = data[indices[i]];
      m = v < m ? v : m;
      M = v > M ? v : M;
    }
    min[j] = m;
    max[j] = M;
  }
  return *this;
}

// Leaf size threshold
static constexpr size_t LEAF_SIZE_THRESHOLD = 256;

BTree::BTree(DataTable* centroids) : centroids(centroids) {
  assert(centroids);
  const size_t numRows = centroids->getNumRows();

  // 1. Initialize the index array (0, 1, 2, ..., numRows-1)
  std::vector<uint32_t> indices(numRows);
  std::iota(indices.begin(), indices.end(), 0);

  // 2. Build the tree recursively
  root = recurse(absl::MakeSpan(indices));
}

/**
 * @brief Recursive function to build the BTree node.
 * The input vector 'indices' is modified in place and partitioned.
 *
 * @param indices The index vector for the current node's elements (passed by move).
 * @return A unique pointer to the newly created BTreeNode.
 */
std::unique_ptr<BTree::BTreeNode> BTree::recurse(absl::Span<uint32_t> indices) {
  auto node = std::make_unique<BTreeNode>();

  AABB aabb;
  aabb.fromCentroids(centroids, indices);

  if (indices.size() <= 256) {
    node->count = indices.size();
    node->aabb = aabb;
    node->indices.assign(indices.begin(), indices.end());
    node->left = nullptr;
    node->right = nullptr;
    return node;
  }

  const int col = aabb.largestAxis();
  const auto& values = centroids->getColumn(col).asSpan<float>();
  const size_t mid = indices.size() / 2;
  quickselect(values, indices, mid);
  auto&& left = recurse(indices.subspan(0, mid));
  auto&& right = recurse(indices.subspan(mid));

  size_t rc = 0;
  rc += left ? left->count : 0;
  rc += right ? right->count : 0;

  node->count = rc;
  node->aabb = aabb;
  node->left = std::move(left);
  node->right = std::move(right);

  return node;
}

}  // namespace splat
