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

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstdint>
#include <memory>

#include "splat/voxel/sparse-octree.h"

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
void getGaussianAABB(const DataTable& extents, const DataTable& dataTable, int index,
                     Eigen::Vector3f& outMin, Eigen::Vector3f& outMax);

/**
 * @brief Check if a Gaussian's AABB overlaps with a given box
 * @param extents DataTable with extent_x, extent_y, extent_z columns
 * @param dataTable DataTable containing position data (x, y, z)
 * @param index Gaussian index
 * @param boxMin Minimum corner of query box
 * @param boxMax Maximum corner of query box
 * @return true if AABBs overlap
 */
bool gaussianOverlapsBox(const DataTable& extents, const DataTable& dataTable, int index,
                         const Eigen::Vector3f& boxMin, const Eigen::Vector3f& boxMax);

}  // namespace splat
