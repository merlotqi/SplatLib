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

#include "splat/voxel/nav-simplify.h"
#include <splat/op/morton_order.h>
#include <splat/utils/logger.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace splat {

/* -------------------------------------------------------------------------- */
/*                          Bitfield grid operations                          */
/* -------------------------------------------------------------------------- */

static void fillDenseSolidGrid(const BlockAccumulator& accumulator, uint32_t* grid,
                               int nx, int ny, int nz) {
  const int stride = nx * ny;

  // Solid blocks
  for (uint32_t morton : accumulator.getSolidBlocks()) {
    auto [bx, by, bz] = mortonToXYZ(morton);
    int baseX = static_cast<int>(bx) << 2;
    int baseY = static_cast<int>(by) << 2;
    int baseZ = static_cast<int>(bz) << 2;
    for (int lz = 0; lz < 4; lz++) {
      int iz = baseZ + lz;
      if (iz >= nz) continue;
      for (int ly = 0; ly < 4; ly++) {
        int iy = baseY + ly;
        if (iy >= ny) continue;
        int rowOff = iz * stride + iy * nx;
        for (int lx = 0; lx < 4; lx++) {
          int ix = baseX + lx;
          if (ix < nx) {
            int idx = rowOff + ix;
            grid[idx >> 5] |= (1u << (idx & 31));
          }
        }
      }
    }
  }

  // Mixed blocks
  auto mixed = accumulator.getMixedBlocks();
  for (size_t i = 0; i < mixed.morton.size(); i++) {
    auto [bx, by, bz] = mortonToXYZ(mixed.morton[i]);
    uint32_t lo = mixed.masks[i * 2];
    uint32_t hi = mixed.masks[i * 2 + 1];
    int baseX = static_cast<int>(bx) << 2;
    int baseY = static_cast<int>(by) << 2;
    int baseZ = static_cast<int>(bz) << 2;
    for (int lz = 0; lz < 4; lz++) {
      int iz = baseZ + lz;
      if (iz >= nz) continue;
      for (int ly = 0; ly < 4; ly++) {
        int iy = baseY + ly;
        if (iy >= ny) continue;
        int rowOff = iz * stride + iy * nx;
        for (int lx = 0; lx < 4; lx++) {
          int bitIdx = lx + (ly << 2) + (lz << 4);
          uint32_t word = bitIdx < 32 ? lo : hi;
          int bit = bitIdx < 32 ? bitIdx : bitIdx - 32;
          if ((word >> bit) & 1) {
            int ix = baseX + lx;
            if (ix < nx) {
              int idx = rowOff + ix;
              grid[idx >> 5] |= (1u << (idx & 31));
            }
          }
        }
      }
    }
  }
}

static void dilateX(const uint32_t* src, uint32_t* dst, int nx, int ny, int nz, int halfExtent) {
  const int stride = nx * ny;
  for (int iz = 0; iz < nz; iz++) {
    for (int iy = 0; iy < ny; iy++) {
      int rowOff = iz * stride + iy * nx;
      int count = 0;
      int winEnd = std::min(halfExtent, nx - 1);
      for (int ix = 0; ix <= winEnd; ix++) {
        int idx = rowOff + ix;
        if ((src[idx >> 5] >> (idx & 31)) & 1) count++;
      }
      for (int ix = 0; ix < nx; ix++) {
        int idx = rowOff + ix;
        if (count > 0) dst[idx >> 5] |= (1u << (idx & 31));
        int exitX = ix - halfExtent;
        if (exitX >= 0) {
          int ei = rowOff + exitX;
          if ((src[ei >> 5] >> (ei & 31)) & 1) count--;
        }
        int enterX = ix + halfExtent + 1;
        if (enterX < nx) {
          int ni = rowOff + enterX;
          if ((src[ni >> 5] >> (ni & 31)) & 1) count++;
        }
      }
    }
  }
}

static void dilateY(const uint32_t* src, uint32_t* dst, int nx, int ny, int nz, int halfExtent) {
  const int stride = nx * ny;
  for (int iz = 0; iz < nz; iz++) {
    int zOff = iz * stride;
    for (int ix = 0; ix < nx; ix++) {
      int count = 0;
      int winEnd = std::min(halfExtent, ny - 1);
      for (int iy = 0; iy <= winEnd; iy++) {
        int idx = zOff + iy * nx + ix;
        if ((src[idx >> 5] >> (idx & 31)) & 1) count++;
      }
      for (int iy = 0; iy < ny; iy++) {
        int idx = zOff + iy * nx + ix;
        if (count > 0) dst[idx >> 5] |= (1u << (idx & 31));
        int exitY = iy - halfExtent;
        if (exitY >= 0) {
          int ei = zOff + exitY * nx + ix;
          if ((src[ei >> 5] >> (ei & 31)) & 1) count--;
        }
        int enterY = iy + halfExtent + 1;
        if (enterY < ny) {
          int ni = zOff + enterY * nx + ix;
          if ((src[ni >> 5] >> (ni & 31)) & 1) count++;
        }
      }
    }
  }
}

static void dilateZ(const uint32_t* src, uint32_t* dst, int nx, int ny, int nz, int halfExtent) {
  const int stride = nx * ny;
  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      int count = 0;
      int winEnd = std::min(halfExtent, nz - 1);
      for (int iz = 0; iz <= winEnd; iz++) {
        int idx = iz * stride + iy * nx + ix;
        if ((src[idx >> 5] >> (idx & 31)) & 1) count++;
      }
      for (int iz = 0; iz < nz; iz++) {
        int idx = iz * stride + iy * nx + ix;
        if (count > 0) dst[idx >> 5] |= (1u << (idx & 31));
        int exitZ = iz - halfExtent;
        if (exitZ >= 0) {
          int ei = exitZ * stride + iy * nx + ix;
          if ((src[ei >> 5] >> (ei & 31)) & 1) count--;
        }
        int enterZ = iz + halfExtent + 1;
        if (enterZ < nz) {
          int ni = enterZ * stride + iy * nx + ix;
          if ((src[ni >> 5] >> (ni & 31)) & 1) count++;
        }
      }
    }
  }
}

static void erodeX(const uint32_t* src, uint32_t* dst, int nx, int ny, int nz, int halfExtent) {
  const int stride = nx * ny;
  for (int iz = 0; iz < nz; iz++) {
    for (int iy = 0; iy < ny; iy++) {
      int rowOff = iz * stride + iy * nx;
      int zeroCount = 0;
      int winEnd = std::min(halfExtent, nx - 1);
      for (int ix = 0; ix <= winEnd; ix++) {
        int idx = rowOff + ix;
        if (!((src[idx >> 5] >> (idx & 31)) & 1)) zeroCount++;
      }
      for (int ix = 0; ix < nx; ix++) {
        int idx = rowOff + ix;
        if (zeroCount == 0) dst[idx >> 5] |= (1u << (idx & 31));
        int exitX = ix - halfExtent;
        if (exitX >= 0) {
          int ei = rowOff + exitX;
          if (!((src[ei >> 5] >> (ei & 31)) & 1)) zeroCount--;
        }
        int enterX = ix + halfExtent + 1;
        if (enterX < nx) {
          int ni = rowOff + enterX;
          if (!((src[ni >> 5] >> (ni & 31)) & 1)) zeroCount++;
        }
      }
    }
  }
}

static void erodeY(const uint32_t* src, uint32_t* dst, int nx, int ny, int nz, int halfExtent) {
  const int stride = nx * ny;
  for (int iz = 0; iz < nz; iz++) {
    int zOff = iz * stride;
    for (int ix = 0; ix < nx; ix++) {
      int zeroCount = 0;
      int winEnd = std::min(halfExtent, ny - 1);
      for (int iy = 0; iy <= winEnd; iy++) {
        int idx = zOff + iy * nx + ix;
        if (!((src[idx >> 5] >> (idx & 31)) & 1)) zeroCount++;
      }
      for (int iy = 0; iy < ny; iy++) {
        int idx = zOff + iy * nx + ix;
        if (zeroCount == 0) dst[idx >> 5] |= (1u << (idx & 31));
        int exitY = iy - halfExtent;
        if (exitY >= 0) {
          int ei = zOff + exitY * nx + ix;
          if (!((src[ei >> 5] >> (ei & 31)) & 1)) zeroCount--;
        }
        int enterY = iy + halfExtent + 1;
        if (enterY < ny) {
          int ni = zOff + enterY * nx + ix;
          if (!((src[ni >> 5] >> (ni & 31)) & 1)) zeroCount++;
        }
      }
    }
  }
}

static void erodeZ(const uint32_t* src, uint32_t* dst, int nx, int ny, int nz, int halfExtent) {
  const int stride = nx * ny;
  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      int zeroCount = 0;
      int winEnd = std::min(halfExtent, nz - 1);
      for (int iz = 0; iz <= winEnd; iz++) {
        int idx = iz * stride + iy * nx + ix;
        if (!((src[idx >> 5] >> (idx & 31)) & 1)) zeroCount++;
      }
      for (int iz = 0; iz < nz; iz++) {
        int idx = iz * stride + iy * nx + ix;
        if (zeroCount == 0) dst[idx >> 5] |= (1u << (idx & 31));
        int exitZ = iz - halfExtent;
        if (exitZ >= 0) {
          int ei = exitZ * stride + iy * nx + ix;
          if (!((src[ei >> 5] >> (ei & 31)) & 1)) zeroCount--;
        }
        int enterZ = iz + halfExtent + 1;
        if (enterZ < nz) {
          int ni = enterZ * stride + iy * nx + ix;
          if (!((src[ni >> 5] >> (ni & 31)) & 1)) zeroCount++;
        }
      }
    }
  }
}

static BlockAccumulator denseGridToAccumulator(const uint32_t* grid, int nx, int ny, int nz, int crop_min_bx,
                                               int crop_min_by, int crop_min_bz, int crop_max_bx, int crop_max_by,
                                               int crop_max_bz) {
  BlockAccumulator acc;
  const int stride = nx * ny;

  for (int bz = crop_min_bz; bz < crop_max_bz; bz++) {
    for (int by = crop_min_by; by < crop_max_by; by++) {
      for (int bx = crop_min_bx; bx < crop_max_bx; bx++) {
        uint32_t lo = 0, hi = 0;
        int base_x = bx << 2;
        int base_y = by << 2;
        int base_z = bz << 2;

        for (int lz = 0; lz < 4; lz++) {
          for (int ly = 0; ly < 4; ly++) {
            for (int lx = 0; lx < 4; lx++) {
              int idx = (base_x + lx) + (base_y + ly) * nx + (base_z + lz) * stride;
              if ((grid[idx >> 5] >> (idx & 31)) & 1) {
                int bit_idx = lx + (ly << 2) + (lz << 4);
                if (bit_idx < 32) {
                  lo |= (1u << bit_idx);
                } else {
                  hi |= (1u << (bit_idx - 32));
                }
              }
            }
          }
        }

        if (lo != 0 || hi != 0) {
          acc.addBlock(xyzToMorton(static_cast<uint32_t>(bx - crop_min_bx), static_cast<uint32_t>(by - crop_min_by),
                                   static_cast<uint32_t>(bz - crop_min_bz)),
                       lo, hi);
        }
      }
    }
  }

  return acc;
}

static bool findNearestFreeCell(const uint32_t* blocked, int seed_ix, int seed_iy, int seed_iz, int nx, int ny, int nz,
                                int stride, int max_radius, int& out_ix, int& out_iy, int& out_iz) {
  for (int r = 1; r <= max_radius; r++) {
    for (int dz = -r; dz <= r; dz++) {
      for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++) {
          if (std::abs(dx) != r && std::abs(dy) != r && std::abs(dz) != r) continue;
          int ix = seed_ix + dx;
          int iy = seed_iy + dy;
          int iz = seed_iz + dz;
          if (ix < 0 || ix >= nx || iy < 0 || iy >= ny || iz < 0 || iz >= nz) continue;
          int idx = ix + iy * nx + iz * stride;
          if (!((blocked[idx >> 5] >> (idx & 31)) & 1)) {
            out_ix = ix;
            out_iy = iy;
            out_iz = iz;
            return true;
          }
        }
      }
    }
  }
  return false;
}

static std::unique_ptr<BlockAccumulator> cloneAccumulator(const BlockAccumulator& src) {
  auto out = std::make_unique<BlockAccumulator>();
  auto mixed = src.getMixedBlocks();
  for (size_t i = 0; i < mixed.morton.size(); ++i) {
    out->addBlock(mixed.morton[i], mixed.masks[i * 2], mixed.masks[i * 2 + 1]);
  }
  for (uint32_t m : src.getSolidBlocks()) {
    out->addBlock(m, SOLID_MASK, SOLID_MASK);
  }
  return out;
}

NavSimplifyResult simplifyForCapsule(const BlockAccumulator& accumulator, const Bounds& grid_bounds,
                                     float voxel_resolution, float capsule_height, float capsule_radius,
                                     const NavSeed& seed) {
  if (!std::isfinite(voxel_resolution) || voxel_resolution <= 0.f) {
    throw std::invalid_argument("nav simplify: voxelResolution must be finite and > 0");
  }
  if (!std::isfinite(capsule_height) || capsule_height <= 0.f) {
    throw std::invalid_argument("nav simplify: capsuleHeight must be finite and > 0");
  }
  if (!std::isfinite(capsule_radius) || capsule_radius < 0.f) {
    throw std::invalid_argument("nav simplify: capsuleRadius must be finite and >= 0");
  }

  const int nx = static_cast<int>(std::lround((grid_bounds.max.x() - grid_bounds.min.x()) / voxel_resolution));
  const int ny = static_cast<int>(std::lround((grid_bounds.max.y() - grid_bounds.min.y()) / voxel_resolution));
  const int nz = static_cast<int>(std::lround((grid_bounds.max.z() - grid_bounds.min.z()) / voxel_resolution));

  if (nx % 4 != 0 || ny % 4 != 0 || nz % 4 != 0) {
    throw std::runtime_error("Grid dimensions must be multiples of 4");
  }

  if (accumulator.count() == 0) {
    return {cloneAccumulator(accumulator), grid_bounds};
  }

  const int total_voxels = nx * ny * nz;
  const int stride = nx * ny;
  const size_t word_count = (static_cast<size_t>(total_voxels) + 31u) >> 5;

  const int kernel_r = static_cast<int>(std::ceil(capsule_radius / voxel_resolution));
  const int y_half_extent = static_cast<int>(std::ceil(capsule_height / (2.f * voxel_resolution)));

  std::vector<uint32_t> bit_a(word_count, 0);
  fillDenseSolidGrid(accumulator, bit_a.data(), nx, ny, nz);

  std::vector<uint32_t> bit_b(word_count, 0);
  dilateX(bit_a.data(), bit_b.data(), nx, ny, nz, kernel_r);
  std::fill(bit_a.begin(), bit_a.end(), 0u);
  dilateZ(bit_b.data(), bit_a.data(), nx, ny, nz, kernel_r);
  std::fill(bit_b.begin(), bit_b.end(), 0u);
  dilateY(bit_a.data(), bit_b.data(), nx, ny, nz, y_half_extent);

  int seed_ix = static_cast<int>(std::floor((seed.x - grid_bounds.min.x()) / voxel_resolution));
  int seed_iy = static_cast<int>(std::floor((seed.y - grid_bounds.min.y()) / voxel_resolution));
  int seed_iz = static_cast<int>(std::floor((seed.z - grid_bounds.min.z()) / voxel_resolution));

  if (seed_ix < 0 || seed_ix >= nx || seed_iy < 0 || seed_iy >= ny || seed_iz < 0 || seed_iz >= nz) {
    LOG_WARN("nav simplify: seed (%f, %f, %f) outside grid, skipping", seed.x, seed.y, seed.z);
    return {cloneAccumulator(accumulator), grid_bounds};
  }

  int seed_idx = seed_ix + seed_iy * nx + seed_iz * stride;
  if ((bit_b[static_cast<size_t>(seed_idx) >> 5] >> (seed_idx & 31)) & 1) {
    const int max_radius = std::max(kernel_r, y_half_extent) * 2;
    int fix = 0, fiy = 0, fiz = 0;
    if (!findNearestFreeCell(bit_b.data(), seed_ix, seed_iy, seed_iz, nx, ny, nz, stride, max_radius, fix, fiy,
                             fiz)) {
      LOG_WARN("nav simplify: seed blocked after dilation, no free cell within %d voxels, skipping", max_radius);
      return {cloneAccumulator(accumulator), grid_bounds};
    }
    seed_ix = fix;
    seed_iy = fiy;
    seed_iz = fiz;
    seed_idx = seed_ix + seed_iy * nx + seed_iz * stride;
  }

  std::fill(bit_a.begin(), bit_a.end(), 0u);

  size_t queue_cap = 1u << std::min(25, static_cast<int>(std::ceil(std::log2(static_cast<double>(total_voxels + 1)))));
  size_t queue_mask = queue_cap - 1;
  std::vector<uint32_t> bfs_queue(queue_cap);
  size_t q_head = 0;
  size_t q_tail = 0;
  size_t queue_size = 0;

  auto enqueue = [&](int n_idx) {
    const size_t w = static_cast<uint32_t>(n_idx) >> 5;
    const uint32_t m = 1u << (n_idx & 31);
    if (!((bit_b[w] | bit_a[w]) & m)) {
      if (queue_size >= queue_cap) {
        const size_t new_cap = queue_cap << 1;
        std::vector<uint32_t> new_queue(new_cap);
        for (size_t i = 0; i < queue_size; i++) {
          new_queue[i] = bfs_queue[(q_head + i) & queue_mask];
        }
        bfs_queue.swap(new_queue);
        queue_cap = new_cap;
        queue_mask = queue_cap - 1;
        q_head = 0;
        q_tail = queue_size;
      }
      bit_a[w] |= m;
      bfs_queue[q_tail] = static_cast<uint32_t>(n_idx);
      q_tail = (q_tail + 1) & queue_mask;
      queue_size++;
    }
  };

  bit_a[static_cast<size_t>(seed_idx) >> 5] |= (1u << (seed_idx & 31));
  bfs_queue[q_tail] = static_cast<uint32_t>(seed_idx);
  q_tail = (q_tail + 1) & queue_mask;
  queue_size++;

  while (queue_size > 0) {
    const int idx = static_cast<int>(bfs_queue[q_head]);
    q_head = (q_head + 1) & queue_mask;
    queue_size--;

    const int ix = idx % nx;
    const int iy = (idx % stride) / nx;
    const int iz = idx / stride;

    if (ix > 0) enqueue(idx - 1);
    if (ix < nx - 1) enqueue(idx + 1);
    if (iy > 0) enqueue(idx - nx);
    if (iy < ny - 1) enqueue(idx + nx);
    if (iz > 0) enqueue(idx - stride);
    if (iz < nz - 1) enqueue(idx + stride);
  }

  for (size_t w = 0; w < word_count; w++) {
    bit_b[w] |= ~bit_a[w];
  }

  const unsigned tail_bits = static_cast<unsigned>(total_voxels) & 31u;
  if (tail_bits != 0) {
    bit_b[word_count - 1] &= (1u << tail_bits) - 1u;
  }

  std::fill(bit_a.begin(), bit_a.end(), 0u);
  erodeX(bit_b.data(), bit_a.data(), nx, ny, nz, kernel_r);

  std::fill(bit_b.begin(), bit_b.end(), 0u);
  erodeZ(bit_a.data(), bit_b.data(), nx, ny, nz, kernel_r);

  std::fill(bit_a.begin(), bit_a.end(), 0u);
  erodeY(bit_b.data(), bit_a.data(), nx, ny, nz, y_half_extent);

  int min_ix = nx, min_iy = ny, min_iz = nz;
  int max_ix = 0, max_iy = 0, max_iz = 0;

  for (int iz = 0; iz < nz; iz++) {
    const int z_off = iz * stride;
    for (int iy = 0; iy < ny; iy++) {
      const int row_off = z_off + iy * nx;
      for (int ix = 0; ix < nx; ix++) {
        const int idx = row_off + ix;
        if (!((bit_a[static_cast<size_t>(idx) >> 5] >> (idx & 31)) & 1)) {
          min_ix = std::min(min_ix, ix);
          max_ix = std::max(max_ix, ix);
          min_iy = std::min(min_iy, iy);
          max_iy = std::max(max_iy, iy);
          min_iz = std::min(min_iz, iz);
          max_iz = std::max(max_iz, iz);
        }
      }
    }
  }

  const int nbx = nx >> 2;
  const int nby = ny >> 2;
  const int nbz = nz >> 2;
  constexpr int kMargin = 1;
  const int crop_min_bx = std::max(0, (min_ix >> 2) - kMargin);
  const int crop_min_by = std::max(0, (min_iy >> 2) - kMargin);
  const int crop_min_bz = std::max(0, (min_iz >> 2) - kMargin);
  const int crop_max_bx = std::min(nbx, (max_ix >> 2) + 1 + kMargin);
  const int crop_max_by = std::min(nby, (max_iy >> 2) + 1 + kMargin);
  const int crop_max_bz = std::min(nbz, (max_iz >> 2) + 1 + kMargin);

  const float block_size = 4.f * voxel_resolution;
  Eigen::Vector3f cropped_min(grid_bounds.min.x() + crop_min_bx * block_size,
                              grid_bounds.min.y() + crop_min_by * block_size,
                              grid_bounds.min.z() + crop_min_bz * block_size);
  Bounds cropped_bounds;
  cropped_bounds.min = cropped_min;
  cropped_bounds.max =
      Eigen::Vector3f(cropped_min.x() + (crop_max_bx - crop_min_bx) * block_size,
                      cropped_min.y() + (crop_max_by - crop_min_by) * block_size,
                      cropped_min.z() + (crop_max_bz - crop_min_bz) * block_size);

  BlockAccumulator dense =
      denseGridToAccumulator(bit_a.data(), nx, ny, nz, crop_min_bx, crop_min_by, crop_min_bz, crop_max_bx, crop_max_by,
                             crop_max_bz);
  return {std::make_unique<BlockAccumulator>(std::move(dense)), cropped_bounds};
}

}  // namespace splat
