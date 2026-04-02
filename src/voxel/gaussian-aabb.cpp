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

#include "splat/voxel/gaussian-aabb.h"

#include <splat/models/data-table.h>

#include <cmath>
#include <limits>

namespace splat {

GaussianExtentsResult computeGaussianExtents(const DataTable& dataTable) {
  const size_t numRows = dataTable.getNumRows();

  // Get column data spans
  const auto& xCol = dataTable.getColumnByName("x");
  const auto& yCol = dataTable.getColumnByName("y");
  const auto& zCol = dataTable.getColumnByName("z");
  const auto& rxCol = dataTable.getColumnByName("rot_1");
  const auto& ryCol = dataTable.getColumnByName("rot_2");
  const auto& rzCol = dataTable.getColumnByName("rot_3");
  const auto& rwCol = dataTable.getColumnByName("rot_0");
  const auto& sxCol = dataTable.getColumnByName("scale_0");
  const auto& syCol = dataTable.getColumnByName("scale_1");
  const auto& szCol = dataTable.getColumnByName("scale_2");

  const auto xSpan = xCol.asSpan<float>();
  const auto ySpan = yCol.asSpan<float>();
  const auto zSpan = zCol.asSpan<float>();
  const auto rxSpan = rxCol.asSpan<float>();
  const auto rySpan = ryCol.asSpan<float>();
  const auto rzSpan = rzCol.asSpan<float>();
  const auto rwSpan = rwCol.asSpan<float>();
  const auto sxSpan = sxCol.asSpan<float>();
  const auto sySpan = syCol.asSpan<float>();
  const auto szSpan = szCol.asSpan<float>();

  // Allocate output arrays
  std::vector<float> extentX(numRows);
  std::vector<float> extentY(numRows);
  std::vector<float> extentZ(numRows);

  // Scene bounds
  Eigen::Vector3f sceneMin(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::infinity());
  Eigen::Vector3f sceneMax(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                           -std::numeric_limits<float>::infinity());

  int invalidCount = 0;

  for (size_t i = 0; i < numRows; i++) {
    // Get Gaussian properties
    Eigen::Vector3f position(xSpan[i], ySpan[i], zSpan[i]);

    // Quaternion: [qx, qy, qz, qw]
    Eigen::Quaternionf rotation(rwSpan[i], rxSpan[i], rySpan[i], rzSpan[i]);
    rotation.normalize();

    // Exponentiate log scale
    Eigen::Vector3f scale(std::exp(sxSpan[i]), std::exp(sySpan[i]), std::exp(szSpan[i]));

    // Create rotation matrix and scale to get the transformation matrix
    Eigen::Matrix3f rotMat = rotation.toRotationMatrix();
    Eigen::Matrix3f scaledMat = rotMat * scale.asDiagonal();

    // Compute AABB by transforming the local AABB (3-sigma box) into world space
    const float sigma3 = 3.0f;
    Eigen::Vector3f localHalfExtents = scale * sigma3;

    // Transform the 8 corners of the local AABB
    Eigen::Vector3f worldMin(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity());
    Eigen::Vector3f worldMax(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity());

    // All 8 combinations of +/- 1 for local corners
    for (int sx = -1; sx <= 1; sx += 2) {
      for (int sy = -1; sy <= 1; sy += 2) {
        for (int sz = -1; sz <= 1; sz += 2) {
          Eigen::Vector3f localCorner(static_cast<float>(sx) * localHalfExtents.x(),
                                      static_cast<float>(sy) * localHalfExtents.y(),
                                      static_cast<float>(sz) * localHalfExtents.z());
          Eigen::Vector3f worldCorner = position + scaledMat * localCorner;
          worldMin = worldMin.cwiseMin(worldCorner);
          worldMax = worldMax.cwiseMax(worldCorner);
        }
      }
    }

    // Compute half-extents
    Eigen::Vector3f halfExtents = (worldMax - worldMin) * 0.5f;
    Eigen::Vector3f center = (worldMax + worldMin) * 0.5f;

    // Validate
    if (!halfExtents.allFinite()) {
      extentX[i] = 0;
      extentY[i] = 0;
      extentZ[i] = 0;
      invalidCount++;
      continue;
    }

    extentX[i] = halfExtents.x();
    extentY[i] = halfExtents.y();
    extentZ[i] = halfExtents.z();

    // Update scene bounds
    sceneMin = sceneMin.cwiseMin(center - halfExtents);
    sceneMax = sceneMax.cwiseMax(center + halfExtents);
  }

  // Create DataTable with extent columns
  auto extentsTable = std::make_unique<DataTable>(std::vector<Column>{Column{"extent_x", std::move(extentX)},
                                                                      Column{"extent_y", std::move(extentY)},
                                                                      Column{"extent_z", std::move(extentZ)}});

  return {std::move(extentsTable), sceneMin, sceneMax, invalidCount};
}

void getGaussianAABB(const DataTable& extents, const DataTable& dataTable, int index, Eigen::Vector3f& outMin,
                     Eigen::Vector3f& outMax) {
  const auto& xCol = dataTable.getColumnByName("x");
  const auto& yCol = dataTable.getColumnByName("y");
  const auto& zCol = dataTable.getColumnByName("z");
  const auto& exCol = extents.getColumnByName("extent_x");
  const auto& eyCol = extents.getColumnByName("extent_y");
  const auto& ezCol = extents.getColumnByName("extent_z");

  outMin = Eigen::Vector3f(xCol.getValue(index) - exCol.getValue(index), yCol.getValue(index) - eyCol.getValue(index),
                           zCol.getValue(index) - ezCol.getValue(index));
  outMax = Eigen::Vector3f(xCol.getValue(index) + exCol.getValue(index), yCol.getValue(index) + eyCol.getValue(index),
                           zCol.getValue(index) + ezCol.getValue(index));
}

bool gaussianOverlapsBox(const DataTable& extents, const DataTable& dataTable, int index, const Eigen::Vector3f& boxMin,
                         const Eigen::Vector3f& boxMax) {
  const auto& xCol = dataTable.getColumnByName("x");
  const auto& yCol = dataTable.getColumnByName("y");
  const auto& zCol = dataTable.getColumnByName("z");
  const auto& exCol = extents.getColumnByName("extent_x");
  const auto& eyCol = extents.getColumnByName("extent_y");
  const auto& ezCol = extents.getColumnByName("extent_z");

  float x = xCol.getValue(index);
  float y = yCol.getValue(index);
  float z = zCol.getValue(index);
  float ex = exCol.getValue(index);
  float ey = eyCol.getValue(index);
  float ez = ezCol.getValue(index);

  // Gaussian AABB
  float gMinX = x - ex, gMaxX = x + ex;
  float gMinY = y - ey, gMaxY = y + ey;
  float gMinZ = z - ez, gMaxZ = z + ez;

  // AABB overlap test
  return !(gMaxX < boxMin.x() || gMinX > boxMax.x() || gMaxY < boxMin.y() || gMinY > boxMax.y() || gMaxZ < boxMin.z() ||
           gMinZ > boxMax.z());
}

}  // namespace splat
