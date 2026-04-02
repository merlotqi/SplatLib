/**
 * CUDA port of splat-transform/lib/voxel/gpu-voxelization.ts (WGSL voxelizeMultiBatch).
 */

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr unsigned kTileSize = 64u;
constexpr unsigned kMaxBlocksPerBatch = 4096u;

struct __align__(16) Gaussian {
  float pos_x, pos_y, pos_z, opacity_logit;
  float rot_w, rot_x, rot_y, rot_z;
  float scale_x, scale_y, scale_z;
  float extent_x, extent_y, extent_z;
  float _pad0, _pad1;
};

struct BatchInfoGpu {
  std::uint32_t index_offset;
  std::uint32_t index_count;
  std::uint32_t num_blocks_x;
  std::uint32_t num_blocks_y;
  std::uint32_t num_blocks_z;
  float block_min_x;
  float block_min_y;
  float block_min_z;
};

static_assert(sizeof(Gaussian) == 64, "Gaussian size");
static_assert(sizeof(BatchInfoGpu) == 32, "BatchInfoGpu size");

__device__ __forceinline__ uint3 morton_to_xyz(unsigned m) {
  return make_uint3((m & 1u) | ((m >> 2u) & 2u), ((m >> 1u) & 1u) | ((m >> 3u) & 2u),
                    ((m >> 2u) & 1u) | ((m >> 4u) & 2u));
}

__device__ __forceinline__ float3 cross3(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ float evaluate_gaussian_for_voxel(float3 voxel_center, float voxel_half_size, const Gaussian& g) {
  const float3 gc = make_float3(g.pos_x, g.pos_y, g.pos_z);
  const float3 diff = make_float3(voxel_center.x - gc.x, voxel_center.y - gc.y, voxel_center.z - gc.z);
  const float3 ext = make_float3(g.extent_x, g.extent_y, g.extent_z);
  if (fabsf(diff.x) > ext.x + voxel_half_size || fabsf(diff.y) > ext.y + voxel_half_size ||
      fabsf(diff.z) > ext.z + voxel_half_size) {
    return 0.f;
  }

  const float3 vmin =
      make_float3(voxel_center.x - voxel_half_size, voxel_center.y - voxel_half_size, voxel_center.z - voxel_half_size);
  const float3 vmax =
      make_float3(voxel_center.x + voxel_half_size, voxel_center.y + voxel_half_size, voxel_center.z + voxel_half_size);
  const float3 closest =
      make_float3(fminf(fmaxf(gc.x, vmin.x), vmax.x), fminf(fmaxf(gc.y, vmin.y), vmax.y),
                  fminf(fmaxf(gc.z, vmin.z), vmax.z));
  float3 closest_diff = make_float3(closest.x - gc.x, closest.y - gc.y, closest.z - gc.z);

  const float3 qxyz = make_float3(-g.rot_x, -g.rot_y, -g.rot_z);
  const float3 cross0 = cross3(qxyz, closest_diff);
  const float3 t = make_float3(2.f * cross0.x, 2.f * cross0.y, 2.f * cross0.z);
  const float3 t2 = cross3(qxyz, t);
  const float3 local_diff = make_float3(closest_diff.x + g.rot_w * t.x + t2.x, closest_diff.y + g.rot_w * t.y + t2.y,
                                        closest_diff.z + g.rot_w * t.z + t2.z);

  const float inv_sx = expf(-g.scale_x);
  const float inv_sy = expf(-g.scale_y);
  const float inv_sz = expf(-g.scale_z);
  const float sx = local_diff.x * inv_sx;
  const float sy = local_diff.y * inv_sy;
  const float sz = local_diff.z * inv_sz;
  const float d2 = sx * sx + sy * sy + sz * sz;
  const float opacity = 1.f / (1.f + expf(-g.opacity_logit));
  return opacity * expf(-0.5f * d2);
}

__global__ void voxelize_multibatch_kernel(float opacity_cutoff, float voxel_resolution,
                                           unsigned int max_blocks_per_batch, const Gaussian* __restrict__ all_gaussians,
                                           const std::uint32_t* __restrict__ indices, unsigned int* __restrict__ results,
                                           const BatchInfoGpu* __restrict__ batch_infos) {
  __shared__ Gaussian shared_gaussians[kTileSize];

  const unsigned batch_idx = blockIdx.z;
  const unsigned flat_block_id = blockIdx.x;
  const unsigned voxel_idx = threadIdx.x;

  const BatchInfoGpu info = batch_infos[batch_idx];
  const unsigned total_blocks = info.num_blocks_x * info.num_blocks_y * info.num_blocks_z;
  if (flat_block_id >= total_blocks) {
    return;
  }

  const unsigned block_x = flat_block_id % info.num_blocks_x;
  const unsigned block_y = (flat_block_id / info.num_blocks_x) % info.num_blocks_y;
  const unsigned block_z = flat_block_id / (info.num_blocks_x * info.num_blocks_y);

  const uint3 local_pos = morton_to_xyz(voxel_idx);
  const float3 block_min = make_float3(info.block_min_x, info.block_min_y, info.block_min_z);
  const float br = voxel_resolution * 4.f;
  const float3 block_offset = make_float3(static_cast<float>(block_x) * br, static_cast<float>(block_y) * br,
                                          static_cast<float>(block_z) * br);
  const float3 voxel_center =
      make_float3(block_min.x + block_offset.x + (static_cast<float>(local_pos.x) + 0.5f) * voxel_resolution,
                  block_min.y + block_offset.y + (static_cast<float>(local_pos.y) + 0.5f) * voxel_resolution,
                  block_min.z + block_offset.z + (static_cast<float>(local_pos.z) + 0.5f) * voxel_resolution);
  const float voxel_half_size = voxel_resolution * 0.5f;

  float total_sigma = 0.f;
  const unsigned num_indices = info.index_count;
  const unsigned num_tiles = (num_indices + kTileSize - 1u) / kTileSize;

  for (unsigned tile = 0u; tile < num_tiles; ++tile) {
    const unsigned load_idx = tile * kTileSize + voxel_idx;
    if (load_idx < num_indices) {
      const std::uint32_t gaussian_idx = indices[info.index_offset + load_idx];
      shared_gaussians[voxel_idx] = all_gaussians[gaussian_idx];
    }
    __syncthreads();

    if (total_sigma < 7.f) {
      const unsigned rem = num_indices - tile * kTileSize;
      const unsigned this_tile_size = rem < kTileSize ? rem : kTileSize;
      for (unsigned c = 0u; c < this_tile_size; ++c) {
        total_sigma += evaluate_gaussian_for_voxel(voxel_center, voxel_half_size, shared_gaussians[c]);
        if (total_sigma >= 7.f) {
          break;
        }
      }
    }
    __syncthreads();
  }

  const float final_opacity = 1.f - expf(-total_sigma);
  const bool is_solid = final_opacity >= opacity_cutoff;
  const unsigned linear_idx = local_pos.z * 16u + local_pos.y * 4u + local_pos.x;
  const unsigned result_base = batch_idx * max_blocks_per_batch * 2u;
  const unsigned word_index = result_base + flat_block_id * 2u + (linear_idx >> 5u);
  const unsigned bit_index = linear_idx & 31u;
  if (is_solid) {
    atomicOr(results + word_index, 1u << bit_index);
  }
}

}  // namespace

extern "C" cudaError_t splat_cuda_voxel_multibatch_launch(cudaStream_t stream, float opacity_cutoff,
                                                          float voxel_resolution, unsigned int max_blocks_per_batch,
                                                          const float* d_gaussians, const std::uint32_t* d_indices,
                                                          unsigned int* d_results, const void* d_batch_infos,
                                                          int num_batches) {
  if (num_batches <= 0) {
    return cudaSuccess;
  }
  const dim3 block(kTileSize, 1, 1);
  const dim3 grid(kMaxBlocksPerBatch, 1, static_cast<unsigned>(num_batches));
  voxelize_multibatch_kernel<<<grid, block, 0, stream>>>(
      opacity_cutoff, voxel_resolution, max_blocks_per_batch, reinterpret_cast<const Gaussian*>(d_gaussians), d_indices,
      d_results, reinterpret_cast<const BatchInfoGpu*>(d_batch_infos));
  return cudaGetLastError();
}
