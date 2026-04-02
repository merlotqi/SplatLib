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

#include "splat/voxel/gaussian-bvh.h"

#include <splat/models/data-table.h>

#include <algorithm>
#include <limits>

namespace splat {

// Partition indices around the k-th largest element using quickselect
static uint32_t quickselect(const std::vector<float>& data, std::vector<uint32_t>& idx, size_t offset, size_t count,
                            size_t k) {
  const auto valAt = [&](size_t p) { return data[idx[offset + p]]; };
  const auto swap = [&](size_t i, size_t j) { std::swap(idx[offset + i], idx[offset + j]); };

  int64_t l = 0;
  int64_t r = static_cast<int64_t>(count) - 1;

  while (true) {
    if (r <= l + 1) {
      if (r == l + 1 && valAt(r) < valAt(l)) swap(l, r);
      return idx[offset + k];
    }

    size_t mid = static_cast<size_t>(l + r) >> 1;
    swap(mid, 1);
    if (valAt(l) > valAt(r)) swap(l, r);
    if (valAt(1) > valAt(r)) swap(1, r);
    if (valAt(l) > valAt(1)) swap(l, 1);

    int64_t i = l + 1;
    int64_t j = r;
    const float pivotVal = valAt(1);

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

    swap(1, j);

    if (static_cast<size_t>(j) >= k) r = j - 1;
    if (static_cast<size_t>(j) <= k) l = i;
  }
}

GaussianBVH::GaussianBVH(const DataTable& dataTable, const DataTable& extents) {
  // Cache column data
  x_ = &dataTable.getColumnByName("x").asVector<float>();
  y_ = &dataTable.getColumnByName("y").asVector<float>();
  z_ = &dataTable.getColumnByName("z").asVector<float>();
  extentX_ = &extents.getColumnByName("extent_x").asVector<float>();
  extentY_ = &extents.getColumnByName("extent_y").asVector<float>();
  extentZ_ = &extents.getColumnByName("extent_z").asVector<float>();

  count_ = dataTable.getNumRows();

  std::vector<uint32_t> indices(static_cast<uint32_t>(count_));
  for (uint32_t i = 0; i < indices.size(); i++) {
    indices[i] = i;
  }

  // Reserve space and build
  nodes_.reserve(count_ * 2);
  buildNode(indices, 0, count_);
}

size_t GaussianBVH::buildNode(std::vector<uint32_t>& indices, size_t offset, size_t count) {
  const size_t nodeIdx = nodes_.size();
  BVHNode node;
  node.count = static_cast<uint32_t>(count);

  // Compute bounds for this node
  float minX = std::numeric_limits<float>::infinity();
  float minY = std::numeric_limits<float>::infinity();
  float minZ = std::numeric_limits<float>::infinity();
  float maxX = -std::numeric_limits<float>::infinity();
  float maxY = -std::numeric_limits<float>::infinity();
  float maxZ = -std::numeric_limits<float>::infinity();

  for (size_t i = 0; i < count; i++) {
    uint32_t idx = indices[offset + i];
    float ex = (*extentX_)[idx], ey = (*extentY_)[idx], ez = (*extentZ_)[idx];
    float px = (*x_)[idx], py = (*y_)[idx], pz = (*z_)[idx];

    float gMinX = px - ex, gMinY = py - ey, gMinZ = pz - ez;
    float gMaxX = px + ex, gMaxY = py + ey, gMaxZ = pz + ez;

    minX = std::min(minX, gMinX);
    minY = std::min(minY, gMinY);
    minZ = std::min(minZ, gMinZ);
    maxX = std::max(maxX, gMaxX);
    maxY = std::max(maxY, gMaxY);
    maxZ = std::max(maxZ, gMaxZ);
  }

  node.bounds = {minX, minY, minZ, maxX, maxY, maxZ};

  // If leaf node
  if (count <= MAX_LEAF_SIZE) {
    node.leftOffset = 0;
    node.rightOffset = 0;
    node.indices.assign(indices.begin() + offset, indices.begin() + offset + count);
    nodes_.push_back(std::move(node));
    return nodeIdx;
  }

  // Find largest axis to split on
  float centroidMinX = std::numeric_limits<float>::infinity(), centroidMaxX = -std::numeric_limits<float>::infinity();
  float centroidMinY = std::numeric_limits<float>::infinity(), centroidMaxY = -std::numeric_limits<float>::infinity();
  float centroidMinZ = std::numeric_limits<float>::infinity(), centroidMaxZ = -std::numeric_limits<float>::infinity();

  for (size_t i = 0; i < count; i++) {
    uint32_t idx = indices[offset + i];
    float px = (*x_)[idx], py = (*y_)[idx], pz = (*z_)[idx];
    centroidMinX = std::min(centroidMinX, px);
    centroidMinY = std::min(centroidMinY, py);
    centroidMinZ = std::min(centroidMinZ, pz);
    centroidMaxX = std::max(centroidMaxX, px);
    centroidMaxY = std::max(centroidMaxY, py);
    centroidMaxZ = std::max(centroidMaxZ, pz);
  }

  const float extX = centroidMaxX - centroidMinX;
  const float extY = centroidMaxY - centroidMinY;
  const float extZ = centroidMaxZ - centroidMinZ;

  // Choose axis with largest extent
  const std::vector<float>* splitAxis = &*x_;
  if (extX >= extY && extX >= extZ) {
    splitAxis = &*x_;
  } else if (extY >= extZ) {
    splitAxis = &*y_;
  } else {
    splitAxis = &*z_;
  }

  // Partition around median
  const size_t mid = count >> 1;
  quickselect(*splitAxis, indices, offset, count, mid);

  // Reserve placeholders
  node.leftOffset = 0;   // Will be filled later
  node.rightOffset = 0;  // Will be filled later
  nodes_.push_back(std::move(node));

  // Recursively build children
  size_t leftIdx = buildNode(indices, offset, mid);
  size_t rightIdx = buildNode(indices, offset + mid, count - mid);

  nodes_[nodeIdx].leftOffset = static_cast<uint32_t>(leftIdx);
  nodes_[nodeIdx].rightOffset = static_cast<uint32_t>(rightIdx);

  // Update root bounds
  rootBounds_ = nodes_[0].bounds;

  return nodeIdx;
}

bool GaussianBVH::boundsOverlap(const BVHBounds& a, float bMinX, float bMinY, float bMinZ, float bMaxX, float bMaxY,
                                float bMaxZ) const {
  return !(a.maxX < bMinX || a.minX > bMaxX || a.maxY < bMinY || a.minY > bMaxY || a.maxZ < bMinZ || a.minZ > bMaxZ);
}

void GaussianBVH::queryNode(const Eigen::Vector3f& min, const Eigen::Vector3f& max, size_t nodeIdx,
                            std::vector<uint32_t>& result) const {
  const auto& node = nodes_[nodeIdx];

  if (!boundsOverlap(node.bounds, min.x(), min.y(), min.z(), max.x(), max.y(), max.z())) {
    return;
  }

  // Leaf node
  if (!node.indices.empty()) {
    for (uint32_t idx : node.indices) {
      float gMinX = (*x_)[idx] - (*extentX_)[idx];
      float gMinY = (*y_)[idx] - (*extentY_)[idx];
      float gMinZ = (*z_)[idx] - (*extentZ_)[idx];
      float gMaxX = (*x_)[idx] + (*extentX_)[idx];
      float gMaxY = (*y_)[idx] + (*extentY_)[idx];
      float gMaxZ = (*z_)[idx] + (*extentZ_)[idx];

      // AABB overlap test
      if (!(gMaxX < min.x() || gMinX > max.x() || gMaxY < min.y() || gMinY > max.y() || gMaxZ < min.z() ||
            gMinZ > max.z())) {
        result.push_back(idx);
      }
    }
    return;
  }

  // Interior node
  if (node.leftOffset > 0) {
    queryNode(min, max, node.leftOffset, result);
  }
  if (node.rightOffset > 0) {
    queryNode(min, max, node.rightOffset, result);
  }
}

std::vector<uint32_t> GaussianBVH::queryOverlapping(const Eigen::Vector3f& boxMin,
                                                    const Eigen::Vector3f& boxMax) const {
  std::vector<uint32_t> result;
  queryNode(boxMin, boxMax, 0, result);
  return result;
}

std::vector<uint32_t> GaussianBVH::queryOverlappingRaw(float minX, float minY, float minZ, float maxX, float maxY,
                                                       float maxZ) const {
  std::vector<uint32_t> result;
  queryNode(Eigen::Vector3f(minX, minY, minZ), Eigen::Vector3f(maxX, maxY, maxZ), 0, result);
  return result;
}

}  // namespace splat
