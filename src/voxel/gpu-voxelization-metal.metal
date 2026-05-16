#include <metal_stdlib>
using namespace metal;

constexpr uint kTileSize = 64u;

struct Gaussian {
  float pos_x;
  float pos_y;
  float pos_z;
  float opacity_logit;
  float rot_w;
  float rot_x;
  float rot_y;
  float rot_z;
  float scale_x;
  float scale_y;
  float scale_z;
  float extent_x;
  float extent_y;
  float extent_z;
  float _pad0;
  float _pad1;
};

struct BatchInfoGpu {
  uint index_offset;
  uint index_count;
  uint num_blocks_x;
  uint num_blocks_y;
  uint num_blocks_z;
  float block_min_x;
  float block_min_y;
  float block_min_z;
};

struct KernelParams {
  float opacity_cutoff;
  float voxel_resolution;
  uint max_blocks_per_batch;
  uint _pad;
};

inline uint3 morton_to_xyz(uint m) {
  return uint3((m & 1u) | ((m >> 2u) & 2u), ((m >> 1u) & 1u) | ((m >> 3u) & 2u),
               ((m >> 2u) & 1u) | ((m >> 4u) & 2u));
}

inline float3 cross3(float3 a, float3 b) {
  return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline float evaluate_gaussian_for_voxel(float3 voxel_center, float voxel_half_size, thread const Gaussian& g) {
  const float3 gc = float3(g.pos_x, g.pos_y, g.pos_z);
  const float3 diff = voxel_center - gc;
  const float3 ext = float3(g.extent_x, g.extent_y, g.extent_z);
  if (fabs(diff.x) > ext.x + voxel_half_size || fabs(diff.y) > ext.y + voxel_half_size ||
      fabs(diff.z) > ext.z + voxel_half_size) {
    return 0.f;
  }

  const float3 vmin = voxel_center - float3(voxel_half_size);
  const float3 vmax = voxel_center + float3(voxel_half_size);
  const float3 closest = clamp(gc, vmin, vmax);
  const float3 closest_diff = closest - gc;

  const float3 qxyz = float3(-g.rot_x, -g.rot_y, -g.rot_z);
  const float3 cross0 = cross3(qxyz, closest_diff);
  const float3 t = 2.f * cross0;
  const float3 t2 = cross3(qxyz, t);
  const float3 local_diff = closest_diff + g.rot_w * t + t2;

  const float inv_sx = exp(-g.scale_x);
  const float inv_sy = exp(-g.scale_y);
  const float inv_sz = exp(-g.scale_z);
  const float sx = local_diff.x * inv_sx;
  const float sy = local_diff.y * inv_sy;
  const float sz = local_diff.z * inv_sz;
  const float d2 = sx * sx + sy * sy + sz * sz;
  const float opacity = 1.f / (1.f + exp(-g.opacity_logit));
  return opacity * exp(-0.5f * d2);
}

kernel void voxelize_multibatch_kernel(device const Gaussian* all_gaussians [[buffer(0)]],
                                       device const uint* indices [[buffer(1)]],
                                       device atomic_uint* results [[buffer(2)]],
                                       device const BatchInfoGpu* batch_infos [[buffer(3)]],
                                       constant KernelParams& params [[buffer(4)]],
                                       uint voxel_idx [[thread_index_in_threadgroup]],
                                       uint3 group_pos [[threadgroup_position_in_grid]]) {
  threadgroup Gaussian shared_gaussians[kTileSize];

  const uint flat_block_id = group_pos.x;
  const uint batch_idx = group_pos.y;

  const BatchInfoGpu info = batch_infos[batch_idx];
  const uint total_blocks = info.num_blocks_x * info.num_blocks_y * info.num_blocks_z;
  if (flat_block_id >= total_blocks) {
    return;
  }

  const uint block_x = flat_block_id % info.num_blocks_x;
  const uint block_y = (flat_block_id / info.num_blocks_x) % info.num_blocks_y;
  const uint block_z = flat_block_id / (info.num_blocks_x * info.num_blocks_y);

  const uint3 local_pos = morton_to_xyz(voxel_idx);
  const float3 block_min = float3(info.block_min_x, info.block_min_y, info.block_min_z);
  const float br = params.voxel_resolution * 4.f;
  const float3 block_offset = float3(float(block_x) * br, float(block_y) * br, float(block_z) * br);
  const float3 voxel_center =
      float3(block_min.x + block_offset.x + (float(local_pos.x) + 0.5f) * params.voxel_resolution,
             block_min.y + block_offset.y + (float(local_pos.y) + 0.5f) * params.voxel_resolution,
             block_min.z + block_offset.z + (float(local_pos.z) + 0.5f) * params.voxel_resolution);
  const float voxel_half_size = params.voxel_resolution * 0.5f;

  float total_sigma = 0.f;
  const uint num_indices = info.index_count;
  const uint num_tiles = (num_indices + kTileSize - 1u) / kTileSize;

  for (uint tile = 0u; tile < num_tiles; ++tile) {
    const uint load_idx = tile * kTileSize + voxel_idx;
    if (load_idx < num_indices) {
      const uint gaussian_idx = indices[info.index_offset + load_idx];
      shared_gaussians[voxel_idx] = all_gaussians[gaussian_idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (total_sigma < 7.f) {
      const uint rem = num_indices - tile * kTileSize;
      const uint this_tile_size = rem < kTileSize ? rem : kTileSize;
      for (uint c = 0u; c < this_tile_size; ++c) {
        total_sigma += evaluate_gaussian_for_voxel(voxel_center, voxel_half_size, shared_gaussians[c]);
        if (total_sigma >= 7.f) {
          break;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float final_opacity = 1.f - exp(-total_sigma);
  if (final_opacity < params.opacity_cutoff) {
    return;
  }

  const uint linear_idx = local_pos.z * 16u + local_pos.y * 4u + local_pos.x;
  const uint result_base = batch_idx * params.max_blocks_per_batch * 2u;
  const uint word_index = result_base + flat_block_id * 2u + (linear_idx >> 5u);
  const uint bit_index = linear_idx & 31u;

  atomic_fetch_or_explicit(&(results[word_index]), (1u << bit_index), memory_order_relaxed);
}
