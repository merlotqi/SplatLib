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

class SplatCloud;

/**
 * @brief Result of computing Gaussian extents
 */
struct GaussianExtentsResult {
  std::unique_ptr<SplatCloud> extents;  ///< SplatCloud containing extent_x, extent_y, extent_z columns
  Eigen::Vector3f sceneMin;             ///< Scene bounds minimum
  Eigen::Vector3f sceneMax;             ///< Scene bounds maximum
  int invalidCount;                     ///< Number of Gaussians skipped due to invalid values
};

/**
 * @brief Compute axis-aligned bounding box half-extents for all Gaussians in a SplatCloud
 *
 * Each Gaussian is an oriented ellipsoid defined by position, rotation (quaternion),
 * and scale (log scale). This function computes the AABB that encloses each
 * rotated ellipsoid and stores only the half-extents.
 *
 * @param dataTable SplatCloud containing Gaussian splat data
 * @return GaussianExtentsResult with extents SplatCloud and scene bounds
 */
GaussianExtentsResult computeGaussianExtents(const SplatCloud& dataTable);

/**
 * @brief Get the AABB for a specific Gaussian
 * @param extents SplatCloud with extent_x, extent_y, extent_z columns
 * @param dataTable SplatCloud containing position data (x, y, z)
 * @param index Gaussian index
 * @param outMin Output Vec3 for minimum corner
 * @param outMax Output Vec3 for maximum corner
 */
void getGaussianAABB(const SplatCloud& extents, const SplatCloud& dataTable, int index, Eigen::Vector3f& outMin,
                     Eigen::Vector3f& outMax);

/**
 * @brief Check if a Gaussian's AABB overlaps with a given box
 * @param extents SplatCloud with extent_x, extent_y, extent_z columns
 * @param dataTable SplatCloud containing position data (x, y, z)
 * @param index Gaussian index
 * @param boxMin Minimum corner of query box
 * @param boxMax Maximum corner of query box
 * @return true if AABBs overlap
 */
bool gaussianOverlapsBox(const SplatCloud& extents, const SplatCloud& dataTable, int index,
                         const Eigen::Vector3f& boxMin, const Eigen::Vector3f& boxMax);

}  // namespace splat
