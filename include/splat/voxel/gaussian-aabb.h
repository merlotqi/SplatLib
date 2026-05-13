/**
 * @file splat/voxel/gaussian-aabb.h
 * @brief Axis-aligned bounds for Gaussians.
 *
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>

/**
 * @file gaussian-aabb.h
 * @brief Compute axis-aligned bounding boxes for Gaussian splats
 */

namespace splat {

class DataTable;

/**
 * @brief Result of computing Gaussian extents
 */
struct GaussianExtentsResult {
  std::unique_ptr<DataTable> extents;  ///< DataTable containing extent_x, extent_y, extent_z columns
  Eigen::Vector3f sceneMin;            ///< Scene bounds minimum
  Eigen::Vector3f sceneMax;            ///< Scene bounds maximum
  int invalidCount;                    ///< Number of Gaussians skipped due to invalid values
};

/**
 * @brief Compute axis-aligned bounding box half-extents for all Gaussians in a DataTable
 *
 * Each Gaussian is an oriented ellipsoid defined by position, rotation (quaternion),
 * and scale (log scale). This function computes the AABB that encloses each
 * rotated ellipsoid and stores only the half-extents.
 *
 * @param dataTable DataTable containing Gaussian splat data
 * @return GaussianExtentsResult with extents DataTable and scene bounds
 */
GaussianExtentsResult computeGaussianExtents(const DataTable& dataTable);

/**
 * @brief Get the AABB for a specific Gaussian
 * @param extents DataTable with extent_x, extent_y, extent_z columns
 * @param dataTable DataTable containing position data (x, y, z)
 * @param index Gaussian index
 * @param outMin Output Vec3 for minimum corner
 * @param outMax Output Vec3 for maximum corner
 */
void getGaussianAABB(const DataTable& extents, const DataTable& dataTable, int index, Eigen::Vector3f& outMin,
                     Eigen::Vector3f& outMax);

/**
 * @brief Check if a Gaussian's AABB overlaps with a given box
 * @param extents DataTable with extent_x, extent_y, extent_z columns
 * @param dataTable DataTable containing position data (x, y, z)
 * @param index Gaussian index
 * @param boxMin Minimum corner of query box
 * @param boxMax Maximum corner of query box
 * @return true if AABBs overlap
 */
bool gaussianOverlapsBox(const DataTable& extents, const DataTable& dataTable, int index, const Eigen::Vector3f& boxMin,
                         const Eigen::Vector3f& boxMax);

}  // namespace splat
