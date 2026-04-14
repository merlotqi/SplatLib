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

#include <splat/utils/logger.h>
#include <splat/voxel/voxel-filter.h>
#include <splat/op/morton_order.h>

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace splat {

namespace {

constexpr uint32_t kFaceX0 = 0x11111111u;
constexpr uint32_t kFaceX3 = 0x88888888u;
constexpr uint32_t kFaceY0 = 0x000F000Fu;
constexpr uint32_t kFaceY3 = 0xF000F000u;
constexpr uint32_t kFaceZ0Lo = 0x0000FFFFu;
constexpr uint32_t kFaceZ3Hi = 0xFFFF0000u;
constexpr uint32_t kSolidMask = 0xFFFFFFFFu;

void addCrossFace(uint32_t nx, uint32_t ny, uint32_t nz, const std::unordered_set<uint32_t>& solid_set,
                  const std::unordered_map<uint32_t, size_t>& mixed_map, const std::vector<uint32_t>& masks,
                  uint32_t our_face_mask, uint32_t adj_face_mask, unsigned shift_amount, bool shift_left,
                  uint32_t cur_lo, uint32_t cur_hi, uint32_t& out_lo, uint32_t& out_hi) {
  const uint32_t adj_morton = xyzToMorton(nx, ny, nz);

  if (solid_set.count(adj_morton)) {
    out_lo = cur_lo | our_face_mask;
    out_hi = cur_hi | our_face_mask;
    return;
  }

  auto it = mixed_map.find(adj_morton);
  if (it == mixed_map.end()) {
    out_lo = cur_lo;
    out_hi = cur_hi;
    return;
  }

  const size_t adj_idx = it->second;
  const uint32_t adj_lo = masks[adj_idx * 2];
  const uint32_t adj_hi = masks[adj_idx * 2 + 1];
  const uint32_t face_lo = adj_lo & adj_face_mask;
  const uint32_t face_hi = adj_hi & adj_face_mask;

  if (shift_left) {
    out_lo = cur_lo | (face_lo << shift_amount);
    out_hi = cur_hi | (face_hi << shift_amount);
  } else {
    out_lo = cur_lo | (face_lo >> shift_amount);
    out_hi = cur_hi | (face_hi >> shift_amount);
  }
}

void addCrossFaceZ(uint32_t nx, uint32_t ny, uint32_t nz, const std::unordered_set<uint32_t>& solid_set,
                   const std::unordered_map<uint32_t, size_t>& mixed_map, const std::vector<uint32_t>& masks,
                   bool plus_z, uint32_t cur_lo, uint32_t cur_hi, uint32_t& out_lo, uint32_t& out_hi) {
  const uint32_t adj_morton = xyzToMorton(nx, ny, nz);

  if (solid_set.count(adj_morton)) {
    if (plus_z) {
      out_lo = cur_lo;
      out_hi = cur_hi | kFaceZ3Hi;
    } else {
      out_lo = cur_lo | kFaceZ0Lo;
      out_hi = cur_hi;
    }
    return;
  }

  auto it = mixed_map.find(adj_morton);
  if (it == mixed_map.end()) {
    out_lo = cur_lo;
    out_hi = cur_hi;
    return;
  }

  const size_t adj_idx = it->second;
  const uint32_t adj_lo = masks[adj_idx * 2];
  const uint32_t adj_hi = masks[adj_idx * 2 + 1];

  if (plus_z) {
    out_lo = cur_lo;
    out_hi = cur_hi | ((adj_lo & kFaceZ0Lo) << 16);
  } else {
    out_lo = cur_lo | ((adj_hi & kFaceZ3Hi) >> 16);
    out_hi = cur_hi;
  }
}

}  // namespace

BlockAccumulator filterAndFillBlocks(const BlockAccumulator& accumulator) {
  const auto mixed = accumulator.getMixedBlocks();
  const auto& solid = accumulator.getSolidBlocks();
  const std::vector<uint32_t>& masks = mixed.masks;

  std::unordered_set<uint32_t> solid_set;
  solid_set.reserve(solid.size() * 2 + 1);
  for (uint32_t m : solid) {
    solid_set.insert(m);
  }

  std::unordered_map<uint32_t, size_t> mixed_map;
  mixed_map.reserve(mixed.morton.size() * 2 + 1);
  for (size_t i = 0; i < mixed.morton.size(); ++i) {
    mixed_map[mixed.morton[i]] = i;
  }

  std::vector<uint32_t> new_masks(masks.size());
  uint64_t voxels_removed = 0;
  uint64_t voxels_filled = 0;

  for (size_t i = 0; i < mixed.morton.size(); ++i) {
    const uint32_t morton = mixed.morton[i];
    const uint32_t orig_lo = masks[i * 2];
    const uint32_t orig_hi = masks[i * 2 + 1];

    const auto xyz = mortonToXYZ(morton);
    const uint32_t bx = xyz[0];
    const uint32_t by = xyz[1];
    const uint32_t bz = xyz[2];

    uint32_t px_lo = (orig_lo >> 1) & ~kFaceX3;
    uint32_t px_hi = (orig_hi >> 1) & ~kFaceX3;

    uint32_t mx_lo = (orig_lo << 1) & ~kFaceX0;
    uint32_t mx_hi = (orig_hi << 1) & ~kFaceX0;

    uint32_t py_lo = (orig_lo >> 4) & ~kFaceY3;
    uint32_t py_hi = (orig_hi >> 4) & ~kFaceY3;

    uint32_t my_lo = (orig_lo << 4) & ~kFaceY0;
    uint32_t my_hi = (orig_hi << 4) & ~kFaceY0;

    uint32_t pz_lo = (orig_lo >> 16) | (orig_hi << 16);
    uint32_t pz_hi = orig_hi >> 16;

    uint32_t mz_lo = orig_lo << 16;
    uint32_t mz_hi = (orig_hi << 16) | (orig_lo >> 16);

    addCrossFace(bx + 1, by, bz, solid_set, mixed_map, masks, kFaceX3, kFaceX0, 3, true, px_lo, px_hi, px_lo, px_hi);
    addCrossFace(bx - 1, by, bz, solid_set, mixed_map, masks, kFaceX0, kFaceX3, 3, false, mx_lo, mx_hi, mx_lo, mx_hi);
    addCrossFace(bx, by + 1, bz, solid_set, mixed_map, masks, kFaceY3, kFaceY0, 12, true, py_lo, py_hi, py_lo, py_hi);
    addCrossFace(bx, by - 1, bz, solid_set, mixed_map, masks, kFaceY0, kFaceY3, 12, false, my_lo, my_hi, my_lo, my_hi);
    addCrossFaceZ(bx, by, bz + 1, solid_set, mixed_map, masks, true, pz_lo, pz_hi, pz_lo, pz_hi);
    addCrossFaceZ(bx, by, bz - 1, solid_set, mixed_map, masks, false, mz_lo, mz_hi, mz_lo, mz_hi);

    const uint32_t neighbor_lo = px_lo | mx_lo | py_lo | my_lo | pz_lo | mz_lo;
    const uint32_t neighbor_hi = px_hi | mx_hi | py_hi | my_hi | pz_hi | mz_hi;
    uint32_t lo = orig_lo & neighbor_lo;
    uint32_t hi = orig_hi & neighbor_hi;

    const uint32_t fill_lo = static_cast<uint32_t>(~lo) & px_lo & mx_lo & py_lo & my_lo & pz_lo & mz_lo;
    const uint32_t fill_hi = static_cast<uint32_t>(~hi) & px_hi & mx_hi & py_hi & my_hi & pz_hi & mz_hi;
    lo |= fill_lo;
    hi |= fill_hi;

    voxels_removed += static_cast<uint64_t>(absl::popcount(orig_lo & ~lo));
    voxels_removed += static_cast<uint64_t>(absl::popcount(orig_hi & ~hi));
    voxels_filled += static_cast<uint64_t>(absl::popcount(lo & ~orig_lo));
    voxels_filled += static_cast<uint64_t>(absl::popcount(hi & ~orig_hi));

    new_masks[i * 2] = lo;
    new_masks[i * 2 + 1] = hi;
  }

  BlockAccumulator result;
  for (size_t i = 0; i < mixed.morton.size(); ++i) {
    result.addBlock(mixed.morton[i], new_masks[i * 2], new_masks[i * 2 + 1]);
  }
  for (uint32_t m : solid) {
    result.addBlock(m, kSolidMask, kSolidMask);
  }

  LOG_INFO("voxel filter: %llu voxels removed, %llu voxels filled",
           static_cast<unsigned long long>(voxels_removed), static_cast<unsigned long long>(voxels_filled));

  return result;
}

}  // namespace splat
