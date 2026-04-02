/***********************************************************************************
 *
 * splat - A C++ library for reading and writing 3D Gaussian Splatting (splat) files.
 *
 * This file is part of splat.
 *
 * splat is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 ***********************************************************************************/

#include <splat/voxel/marching-cubes.h>

#include <splat/voxel/sparse-octree.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace splat {
namespace marching_cubes_tables {
extern const std::uint16_t kEdgeTable[256];
extern const std::vector<std::vector<int>> kTriTable;
}  // namespace marching_cubes_tables

namespace {

bool isVoxelSet(std::uint32_t lo, std::uint32_t hi, int lx, int ly, int lz) {
  const int bit_idx = lx + ly * 4 + lz * 16;
  if (bit_idx < 32) {
    return (lo & (1u << bit_idx)) != 0;
  }
  return (hi & (1u << (bit_idx - 32))) != 0;
}

}  // namespace

MarchingCubesMesh marchingCubes(const BlockAccumulator& accumulator, const Bounds& gridBounds,
                                float voxelResolution) {
  BlockAccumulator::MixedBlocks mixed = accumulator.getMixedBlocks();
  const std::vector<std::uint32_t>& solid = accumulator.getSolidBlocks();

  std::unordered_map<std::uint32_t, std::int32_t> block_map;
  block_map.reserve(mixed.morton.size() + solid.size());
  for (size_t i = 0; i < mixed.morton.size(); ++i) {
    block_map[mixed.morton[i]] = static_cast<std::int32_t>(i);
  }
  for (std::uint32_t m : solid) {
    block_map[m] = -1;
  }

  auto is_occupied = [&](int vx, int vy, int vz) -> bool {
    if (vx < 0 || vy < 0 || vz < 0) {
      return false;
    }
    const int bx = vx >> 2;
    const int by = vy >> 2;
    const int bz = vz >> 2;
    std::uint32_t key = xyzToMorton(static_cast<std::uint32_t>(bx), static_cast<std::uint32_t>(by),
                                     static_cast<std::uint32_t>(bz));
    auto it = block_map.find(key);
    if (it == block_map.end()) {
      return false;
    }
    if (it->second < 0) {
      return true;
    }
    size_t idx = static_cast<size_t>(it->second);
    std::uint32_t lo = mixed.masks[idx * 2];
    std::uint32_t hi = mixed.masks[idx * 2 + 1];
    return isVoxelSet(lo, hi, vx & 3, vy & 3, vz & 3);
  };

  std::unordered_set<std::uint32_t> block_set;
  block_set.reserve(mixed.morton.size() + solid.size());
  for (std::uint32_t m : mixed.morton) {
    block_set.insert(m);
  }
  for (std::uint32_t m : solid) {
    block_set.insert(m);
  }

  std::vector<float> positions;
  std::vector<std::uint32_t> indices;
  positions.reserve(4096);
  indices.reserve(4096);

  const float origin_x = gridBounds.min.x();
  const float origin_y = gridBounds.min.y();
  const float origin_z = gridBounds.min.z();

  const int stride_x =
      static_cast<int>(std::lround((gridBounds.max.x() - gridBounds.min.x()) / voxelResolution)) + 3;
  const int stride_xy =
      stride_x * (static_cast<int>(std::lround((gridBounds.max.y() - gridBounds.min.y()) / voxelResolution)) + 3);

  std::unordered_map<std::uint64_t, std::uint32_t> vertex_map;

  auto get_vertex = [&](int vx, int vy, int vz, int axis) -> std::uint32_t {
    const std::int64_t key =
        (static_cast<std::int64_t>(vx + 1) + static_cast<std::int64_t>(vy + 1) * stride_x +
         static_cast<std::int64_t>(vz + 1) * stride_xy) *
            3 +
        axis;
    const std::uint64_t ukey = static_cast<std::uint64_t>(key);
    auto it = vertex_map.find(ukey);
    if (it != vertex_map.end()) {
      return it->second;
    }
    std::uint32_t idx = static_cast<std::uint32_t>(positions.size() / 3u);
    float px = origin_x + static_cast<float>(vx) * voxelResolution;
    float py = origin_y + static_cast<float>(vy) * voxelResolution;
    float pz = origin_z + static_cast<float>(vz) * voxelResolution;
    if (axis == 0) {
      px += voxelResolution * 0.5f;
    } else if (axis == 1) {
      py += voxelResolution * 0.5f;
    } else {
      pz += voxelResolution * 0.5f;
    }
    positions.push_back(px);
    positions.push_back(py);
    positions.push_back(pz);
    vertex_map.emplace(ukey, idx);
    return idx;
  };

  std::vector<std::uint32_t> all_mortons;
  all_mortons.reserve(block_set.size());
  for (std::uint32_t m : block_set) {
    all_mortons.push_back(m);
  }

  for (std::uint32_t morton : all_mortons) {
    std::array<std::uint32_t, 3> xyz = mortonToXYZ(morton);
    std::uint32_t bx = xyz[0];
    std::uint32_t by = xyz[1];
    std::uint32_t bz = xyz[2];

    for (int lz = -1; lz < 4; ++lz) {
      for (int ly = -1; ly < 4; ++ly) {
        for (int lx = -1; lx < 4; ++lx) {
          const int vx = static_cast<int>(bx) * 4 + lx;
          const int vy = static_cast<int>(by) * 4 + ly;
          const int vz = static_cast<int>(bz) * 4 + lz;

          const int owner_bx = vx >> 2;
          const int owner_by = vy >> 2;
          const int owner_bz = vz >> 2;
          if (owner_bx != static_cast<int>(bx) || owner_by != static_cast<int>(by) ||
              owner_bz != static_cast<int>(bz)) {
            if (owner_bx >= 0 && owner_by >= 0 && owner_bz >= 0) {
              std::uint32_t om = xyzToMorton(static_cast<std::uint32_t>(owner_bx),
                                             static_cast<std::uint32_t>(owner_by),
                                             static_cast<std::uint32_t>(owner_bz));
              if (block_set.count(om) != 0u) {
                continue;
              }
            }
          }

          const int c0 = is_occupied(vx, vy, vz) ? 1 : 0;
          const int c1 = is_occupied(vx + 1, vy, vz) ? 1 : 0;
          const int c2 = is_occupied(vx + 1, vy + 1, vz) ? 1 : 0;
          const int c3 = is_occupied(vx, vy + 1, vz) ? 1 : 0;
          const int c4 = is_occupied(vx, vy, vz + 1) ? 1 : 0;
          const int c5 = is_occupied(vx + 1, vy, vz + 1) ? 1 : 0;
          const int c6 = is_occupied(vx + 1, vy + 1, vz + 1) ? 1 : 0;
          const int c7 = is_occupied(vx, vy + 1, vz + 1) ? 1 : 0;

          const int cube_index = c0 | (c1 << 1) | (c2 << 2) | (c3 << 3) | (c4 << 4) | (c5 << 5) | (c6 << 6) | (c7 << 7);
          if (cube_index == 0 || cube_index == 255) {
            continue;
          }

          const std::uint16_t edges = marching_cubes_tables::kEdgeTable[static_cast<size_t>(cube_index)];
          if (edges == 0) {
            continue;
          }

          std::uint32_t edge_verts[12];
          if (edges & 1) {
            edge_verts[0] = get_vertex(vx, vy, vz, 0);
          }
          if (edges & 2) {
            edge_verts[1] = get_vertex(vx + 1, vy, vz, 1);
          }
          if (edges & 4) {
            edge_verts[2] = get_vertex(vx, vy + 1, vz, 0);
          }
          if (edges & 8) {
            edge_verts[3] = get_vertex(vx, vy, vz, 1);
          }
          if (edges & 16) {
            edge_verts[4] = get_vertex(vx, vy, vz + 1, 0);
          }
          if (edges & 32) {
            edge_verts[5] = get_vertex(vx + 1, vy, vz + 1, 1);
          }
          if (edges & 64) {
            edge_verts[6] = get_vertex(vx, vy + 1, vz + 1, 0);
          }
          if (edges & 128) {
            edge_verts[7] = get_vertex(vx, vy, vz + 1, 1);
          }
          if (edges & 256) {
            edge_verts[8] = get_vertex(vx, vy, vz, 2);
          }
          if (edges & 512) {
            edge_verts[9] = get_vertex(vx + 1, vy, vz, 2);
          }
          if (edges & 1024) {
            edge_verts[10] = get_vertex(vx + 1, vy + 1, vz, 2);
          }
          if (edges & 2048) {
            edge_verts[11] = get_vertex(vx, vy + 1, vz, 2);
          }

          const std::vector<int>& tri_row = marching_cubes_tables::kTriTable[static_cast<size_t>(cube_index)];
          for (size_t t = 0; t + 2 < tri_row.size(); t += 3) {
            indices.push_back(edge_verts[static_cast<size_t>(tri_row[t])]);
            indices.push_back(edge_verts[static_cast<size_t>(tri_row[t + 2])]);
            indices.push_back(edge_verts[static_cast<size_t>(tri_row[t + 1])]);
          }
        }
      }
    }
  }

  MarchingCubesMesh mesh;
  mesh.positions = std::move(positions);
  mesh.indices = std::move(indices);
  return mesh;
}

}  // namespace splat
