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

#pragma once

#include <cstdint>
#include <vector>

namespace splat {

class DataTable;

/**
 * @brief One batch in a multi-batch GPU voxelization dispatch (matches TS BatchSpec).
 */
struct BatchSpec {
  uint32_t index_offset = 0;   ///< Offset into concatenated Gaussian index buffer
  uint32_t index_count = 0;    ///< Number of indices for this batch
  float block_min_x = 0.f;     ///< World-space minimum corner of first block
  float block_min_y = 0.f;
  float block_min_z = 0.f;
  uint32_t num_blocks_x = 0;
  uint32_t num_blocks_y = 0;
  uint32_t num_blocks_z = 0;
};

/**
 * @brief Raw GPU readback: interleaved lo/hi u32 masks per block (matches TS MultiBatchResult).
 */
struct MultiBatchResult {
  std::vector<uint32_t> masks;
  uint32_t max_blocks_per_batch = 4096;
};

/**
 * @brief True if at least one CUDA device is visible (cudaGetDeviceCount).
 */
bool gpuVoxelizationIsAvailable() noexcept;

/**
 * @brief CUDA multi-batch Gaussian voxelization (port of splat-transform GpuVoxelization / WGSL).
 */
class GpuVoxelization {
 public:
  static constexpr int kFloatsPerGaussian = 16;
  static constexpr uint32_t kMaxBlocksPerBatch = 4096;
  static constexpr int kBatchInfoU32s = 8;
  static constexpr int kNumSlots = 2;

  explicit GpuVoxelization(int cuda_device_index = 0);
  GpuVoxelization(const GpuVoxelization&) = delete;
  GpuVoxelization& operator=(const GpuVoxelization&) = delete;
  GpuVoxelization(GpuVoxelization&& other) noexcept;
  GpuVoxelization& operator=(GpuVoxelization&& other) noexcept;
  ~GpuVoxelization();

  void uploadAllGaussians(const DataTable& data_table, const DataTable& extents);
  MultiBatchResult submitMultiBatch(int slot_index, const std::vector<uint32_t>& concatenated_indices,
                                    size_t total_indices, const std::vector<BatchSpec>& batches,
                                    float voxel_resolution, float opacity_cutoff);

  int numGaussians() const noexcept { return num_gaussians_; }
  int cudaDeviceIndex() const noexcept { return cuda_device_index_; }

 private:
  int cuda_device_index_ = 0;
  void* stream_ = nullptr;  // cudaStream_t
  float* d_gaussians_ = nullptr;
  size_t gaussian_buffer_floats_ = 0;

  struct SlotBuffers {
    uint32_t* d_indices = nullptr;
    unsigned int* d_results = nullptr;
    void* d_batch_infos = nullptr;
    size_t index_cap = 0;
    size_t results_byte_cap = 0;
    size_t batch_info_bytes_cap = 0;
  };
  SlotBuffers slots_[2];

  int num_gaussians_ = 0;

  void freeSlot(SlotBuffers& s);
  void destroyStream();
};

}  // namespace splat
