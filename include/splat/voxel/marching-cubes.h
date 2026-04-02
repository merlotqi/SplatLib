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

#include <cstdint>
#include <vector>

#include "splat/voxel/sparse-octree.h"

/**
 * @file marching-cubes.h
 * @brief Marching cubes surface extraction from voxel data
 */

namespace splat {

/**
 * @brief Result of marching cubes surface extraction
 */
struct MarchingCubesMesh {
  std::vector<float> positions;      ///< Vertex positions (3 floats per vertex)
  std::vector<uint32_t> indices;     ///< Triangle indices (3 indices per triangle)
};

/**
 * @brief Extract a triangle mesh from a BlockAccumulator using marching cubes
 *
 * Each voxel is treated as a cell in the marching cubes grid. Corner values
 * are binary (0 = empty, 1 = occupied) with a 0.5 threshold. Vertices are
 * placed at edge midpoints, producing a mesh that follows voxel boundaries.
 *
 * @param accumulator Voxel block data after filtering
 * @param gridBounds Grid bounds aligned to block boundaries
 * @param voxelResolution Size of each voxel in world units
 * @return Mesh with positions and indices
 */
MarchingCubesMesh marchingCubes(const BlockAccumulator& accumulator, const Bounds& gridBounds,
                                float voxelResolution);

}  // namespace splat
