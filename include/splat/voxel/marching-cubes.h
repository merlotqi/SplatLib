/**
 * @file splat/voxel/marching-cubes.h
 * @brief Extract isosurface mesh from scalar fields.
 *
 * References:
 * - [Marching cubes](https://paulbourke.net/geometry/polygonise/)
 */
 
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
