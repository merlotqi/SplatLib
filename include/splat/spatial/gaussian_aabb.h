/**
 * @file gaussian_aabb.h
 * @brief Axis-Aligned Bounding Box calculation for Gaussian splats.
 *
 * This file provides functions for computing bounding volumes and scene
 * extents for 3D Gaussian point cloud datasets.
 */

#pragma once

#include <splat/models/data-table.h>

#include <Eigen/Dense>

namespace splat {

struct GaussianExtentsResult {
  /**
   * DataTable containing extent_x, extent_y, extent_z columns.
   * To compute AABB for Gaussian i:
   *   minX = x[i] - extent_x[i], maxX = x[i] + extent_x[i]
   *   minY = y[i] - extent_y[i], maxY = y[i] + extent_y[i]
   *   minZ = z[i] - extent_z[i], maxZ = z[i] + extent_z[i]
   */
  std::unique_ptr<DataTable> extents{nullptr};

  /** Scene bounds (union of all Gaussian AABBs) */
  struct {
    Eigen::Vector3f min;
    Eigen::Vector3f max;
  } sceneBounds;
  /** Number of Gaussians skipped due to invalid values */
  size_t invalidCount{0};
};

/**
 * Compute axis-aligned bounding box half-extents for all Gaussians in a DataTable.
 *
 * Each Gaussian is an oriented ellipsoid defined by position, rotation (quaternion),
 * and scale (log scale). This function computes the AABB that encloses each
 * rotated ellipsoid and stores only the half-extents. The full AABB can be
 * reconstructed at runtime using: min = position - extent, max = position + extent.
 *
 * @param dataTable - DataTable containing Gaussian splat data
 * @returns GaussianExtentsResult with extents DataTable and scene bounds
 */
GaussianExtentsResult computeGaussianExtents(const DataTable* dataTable);

}  // namespace splat
