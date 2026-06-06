#include <absl/types/span.h>
#include <assert.h>
#include <splat/models/splatcloud.h>
#include <splat/spatial/gaussian_bvh.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>

#include "Eigen/src/Core/Matrix.h"

namespace splat {

extern uint32_t quickselect(absl::Span<const float> data, absl::Span<uint32_t> idx, size_t k_in);

GaussianBVH::GaussianBVH(const SplatCloud* dataTable, const SplatCloud* extents) {
  assert(dataTable);
  assert(extents);

  x_ = dataTable->getColumnByName("x").asSpan<float>();
  y_ = dataTable->getColumnByName("y").asSpan<float>();
  z_ = dataTable->getColumnByName("z").asSpan<float>();

  extentX_ = extents->getColumnByName("extent_x").asSpan<float>();
  extentY_ = extents->getColumnByName("extent_y").asSpan<float>();
  extentZ_ = extents->getColumnByName("extent_z").asSpan<float>();

  const size_t numRows = dataTable->getNumRows();
  std::vector<uint32_t> indices(numRows);
  std::iota(indices.begin(), indices.end(), 0);

  root_ = buildNode(absl::MakeSpan(indices));
}

std::vector<uint32_t> GaussianBVH::queryOverlapping(const Eigen::Vector3f& boxMin, const Eigen::Vector3f& boxMax) {
  std::vector<uint32_t> result;

  queryNode(root_.get(), boxMin.x(), boxMin.y(), boxMin.z(), boxMax.x(), boxMax.y(), boxMax.z(), result);

  return result;
}

GaussianBVH::BVHBounds GaussianBVH::computeBound(absl::Span<uint32_t> indices) {
  float minX = std::numeric_limits<float>::infinity();
  float minY = std::numeric_limits<float>::infinity();
  float minZ = std::numeric_limits<float>::infinity();
  float maxX = -std::numeric_limits<float>::infinity();
  float maxY = -std::numeric_limits<float>::infinity();
  float maxZ = -std::numeric_limits<float>::infinity();

  for (size_t i = 0; i < indices.size(); ++i) {
    const auto idx = indices[i];
    const auto gMinX = x_[idx] - extentX_[idx];
    const auto gMinY = y_[idx] - extentY_[idx];
    const auto gMinZ = z_[idx] - extentZ_[idx];
    const auto gMaxX = x_[idx] + extentX_[idx];
    const auto gMaxY = y_[idx] + extentY_[idx];
    const auto gMaxZ = z_[idx] + extentZ_[idx];

    if (gMinX < minX) minX = gMinX;
    if (gMinY < minY) minY = gMinY;
    if (gMinZ < minZ) minZ = gMinZ;
    if (gMaxX > maxX) maxX = gMaxX;
    if (gMaxY > maxY) maxY = gMaxY;
    if (gMaxZ > maxZ) maxZ = gMaxZ;
  }

  return {Eigen::Vector3f(minX, minY, minZ), Eigen::Vector3f(maxX, maxY, maxZ)};
}

std::unique_ptr<GaussianBVH::BVHNode> GaussianBVH::buildNode(absl::Span<uint32_t> indices) {
  auto bounds = computeBound(indices);

  // Create leaf node if small enough
  if (indices.size() <= MAX_LEAF_SIZE) {
    return std::make_unique<BVHNode>(indices.size(), bounds, std::vector<uint32_t>(indices.begin(), indices.end()));
  }

  // Find the largest axis to split on (based on centroid positions for better balance)
  float centroidMinX = std::numeric_limits<float>::infinity(), centroidMaxX = -std::numeric_limits<float>::infinity();
  float centroidMinY = std::numeric_limits<float>::infinity(), centroidMaxY = -std::numeric_limits<float>::infinity();
  float centroidMinZ = std::numeric_limits<float>::infinity(), centroidMaxZ = -std::numeric_limits<float>::infinity();

  for (size_t i = 0; i < indices.size(); i++) {
    const auto idx = indices[i];
    const auto px = x_[idx];
    const auto py = y_[idx];
    const auto pz = z_[idx];

    if (px < centroidMinX) centroidMinX = px;
    if (px > centroidMaxX) centroidMaxX = px;
    if (py < centroidMinY) centroidMinY = py;
    if (py > centroidMaxY) centroidMaxY = py;
    if (pz < centroidMinZ) centroidMinZ = pz;
    if (pz > centroidMaxZ) centroidMaxZ = pz;
  }

  const auto extX = centroidMaxX - centroidMinX;
  const auto extY = centroidMaxY - centroidMinY;
  const auto extZ = centroidMaxZ - centroidMinZ;

  // Choose axis with largest extent
  absl::Span<const float> splitAxis;
  if (extX >= extY && extX >= extZ) {
    splitAxis = x_;
  } else if (extY >= extZ) {
    splitAxis = y_;
  } else {
    splitAxis = z_;
  }

  // Partition around median
  const size_t mid = indices.size() / 2;
  quickselect(splitAxis, indices, mid);

  auto&& left = buildNode(indices.subspan(0, mid));
  auto&& right = buildNode(indices.subspan(mid));

  size_t rc = 0;
  rc += left ? left->count : 0;
  rc += right ? right->count : 0;

  auto node = std::make_unique<BVHNode>();
  node->count = rc;
  node->bounds = bounds;
  node->left = std::move(left);
  node->right = std::move(right);

  return node;
}

void GaussianBVH::queryNode(const BVHNode* node, float minX, float minY, float minZ, float maxX, float maxY, float maxZ,
                            std::vector<uint32_t>& result) {
  auto boundsOverlap = [](const BVHBounds& a, float bMinX, float bMinY, float bMinZ, float bMaxX, float bMaxY,
                          float bMaxZ) -> bool {
    return !(a.max.x() < bMinX || a.min.x() > bMaxX || a.max.y() < bMinY || a.min.y() > bMaxY || a.max.z() < bMinZ ||
             a.min.z() > bMaxZ);
  };

  // Early exit if node bounds don't overlap query box
  if (!boundsOverlap(node->bounds, minX, minY, minZ, maxX, maxY, maxZ)) {
    return;
  }

  // Leaf node: check each Gaussian individually
  if (!node->indices.empty()) {
    for (size_t i = 0; i < node->indices.size(); i++) {
      const auto idx = node->indices[i];
      const auto gMinX = x_[idx] - extentX_[idx];
      const auto gMinY = y_[idx] - extentY_[idx];
      const auto gMinZ = z_[idx] - extentZ_[idx];
      const auto gMaxX = x_[idx] + extentX_[idx];
      const auto gMaxY = y_[idx] + extentY_[idx];
      const auto gMaxZ = z_[idx] + extentZ_[idx];

      // Check overlap
      if (!(gMaxX < minX || gMinX > maxX || gMaxY < minY || gMinY > maxY || gMaxZ < minZ || gMinZ > maxZ)) {
        result.push_back(idx);
      }
    }
    return;
  }

  // Interior node: recurse into children
  if (node->left) {
    queryNode(node->left.get(), minX, minY, minZ, maxX, maxY, maxZ, result);
  }
  if (node->right) {
    queryNode(node->right.get(), minX, minY, minZ, maxX, maxY, maxZ, result);
  }
}

}  // namespace splat
