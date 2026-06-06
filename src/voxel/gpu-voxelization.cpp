#include <cuda_runtime.h>
#include <splat/models/splatcloud.h>
#include <splat/voxel/gpu-voxelization.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace splat {

namespace {

struct BatchInfoHost {
  std::uint32_t index_offset;
  std::uint32_t index_count;
  std::uint32_t num_blocks_x;
  std::uint32_t num_blocks_y;
  std::uint32_t num_blocks_z;
  float block_min_x;
  float block_min_y;
  float block_min_z;
};

static_assert(sizeof(BatchInfoHost) == 32, "BatchInfoHost layout must match gpu-voxelization-kernels.cu");

extern "C" cudaError_t splat_cuda_voxel_multibatch_launch(cudaStream_t stream, float opacity_cutoff,
                                                          float voxel_resolution, unsigned int max_blocks_per_batch,
                                                          const float* d_gaussians, const std::uint32_t* d_indices,
                                                          unsigned int* d_results, const void* d_batch_infos,
                                                          int num_batches);

static void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

}  // namespace

bool gpuVoxelizationIsAvailable() noexcept {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    return false;
  }
  return n > 0;
}

GpuVoxelization::GpuVoxelization(int cuda_device_index) : cuda_device_index_(cuda_device_index) {
  int count = 0;
  cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (count == 0) {
    throw std::runtime_error("GpuVoxelization: no CUDA devices");
  }
  if (cuda_device_index_ < 0 || cuda_device_index_ >= count) {
    throw std::runtime_error("GpuVoxelization: invalid cuda_device_index");
  }
  cuda_check(cudaSetDevice(cuda_device_index_), "cudaSetDevice");
  cudaStream_t stream = nullptr;
  cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
  stream_ = stream;
}

void GpuVoxelization::freeSlot(SlotBuffers& s) {
  cudaFree(s.d_indices);
  cudaFree(s.d_results);
  cudaFree(s.d_batch_infos);
  s.d_indices = nullptr;
  s.d_results = nullptr;
  s.d_batch_infos = nullptr;
  s.index_cap = 0;
  s.results_byte_cap = 0;
  s.batch_info_bytes_cap = 0;
}

void GpuVoxelization::destroyStream() {
  if (stream_) {
    cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    stream_ = nullptr;
  }
}

GpuVoxelization::~GpuVoxelization() {
  if (stream_) {
    cudaSetDevice(cuda_device_index_);
    freeSlot(slots_[0]);
    freeSlot(slots_[1]);
    cudaFree(d_gaussians_);
    destroyStream();
  }
}

GpuVoxelization::GpuVoxelization(GpuVoxelization&& o) noexcept
    : cuda_device_index_(o.cuda_device_index_),
      stream_(o.stream_),
      d_gaussians_(o.d_gaussians_),
      gaussian_buffer_floats_(o.gaussian_buffer_floats_),
      num_gaussians_(o.num_gaussians_) {
  slots_[0] = o.slots_[0];
  slots_[1] = o.slots_[1];
  o.stream_ = nullptr;
  o.d_gaussians_ = nullptr;
  o.gaussian_buffer_floats_ = 0;
  o.num_gaussians_ = 0;
  o.slots_[0] = {};
  o.slots_[1] = {};
}

GpuVoxelization& GpuVoxelization::operator=(GpuVoxelization&& o) noexcept {
  if (this != &o) {
    if (stream_) {
      cudaSetDevice(cuda_device_index_);
      freeSlot(slots_[0]);
      freeSlot(slots_[1]);
      cudaFree(d_gaussians_);
      destroyStream();
    }
    cuda_device_index_ = o.cuda_device_index_;
    stream_ = o.stream_;
    d_gaussians_ = o.d_gaussians_;
    gaussian_buffer_floats_ = o.gaussian_buffer_floats_;
    num_gaussians_ = o.num_gaussians_;
    slots_[0] = o.slots_[0];
    slots_[1] = o.slots_[1];
    o.stream_ = nullptr;
    o.d_gaussians_ = nullptr;
    o.gaussian_buffer_floats_ = 0;
    o.num_gaussians_ = 0;
    o.slots_[0] = {};
    o.slots_[1] = {};
  }
  return *this;
}

void GpuVoxelization::uploadAllGaussians(const SplatCloud& data_table, const SplatCloud& extents) {
  cudaSetDevice(cuda_device_index_);
  const size_t n = data_table.getNumRows();
  if (extents.getNumRows() != n) {
    throw std::invalid_argument("GpuVoxelization::uploadAllGaussians: extents row count must match data_table");
  }

  const auto& x = data_table.getColumnByName("x").asVector<float>();
  const auto& y = data_table.getColumnByName("y").asVector<float>();
  const auto& z = data_table.getColumnByName("z").asVector<float>();
  const auto& opacity = data_table.getColumnByName("opacity").asVector<float>();
  const auto& rot_w = data_table.getColumnByName("rot_0").asVector<float>();
  const auto& rot_x = data_table.getColumnByName("rot_1").asVector<float>();
  const auto& rot_y = data_table.getColumnByName("rot_2").asVector<float>();
  const auto& rot_z = data_table.getColumnByName("rot_3").asVector<float>();
  const auto& scale_x = data_table.getColumnByName("scale_0").asVector<float>();
  const auto& scale_y = data_table.getColumnByName("scale_1").asVector<float>();
  const auto& scale_z = data_table.getColumnByName("scale_2").asVector<float>();
  const auto& extent_x = extents.getColumnByName("extent_x").asVector<float>();
  const auto& extent_y = extents.getColumnByName("extent_y").asVector<float>();
  const auto& extent_z = extents.getColumnByName("extent_z").asVector<float>();

  const size_t need_floats = n * static_cast<size_t>(kFloatsPerGaussian);
  if (need_floats > gaussian_buffer_floats_) {
    cudaFree(d_gaussians_);
    d_gaussians_ = nullptr;
    gaussian_buffer_floats_ = 0;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_gaussians_), need_floats * sizeof(float)),
               "cudaMalloc gaussians");
    gaussian_buffer_floats_ = need_floats;
  }

  std::vector<float> host(need_floats);
  for (size_t i = 0; i < n; ++i) {
    const size_t o = i * static_cast<size_t>(kFloatsPerGaussian);
    host[o + 0] = x[i];
    host[o + 1] = y[i];
    host[o + 2] = z[i];
    host[o + 3] = opacity[i];
    float qw = rot_w[i], qx = rot_x[i], qy = rot_y[i], qz = rot_z[i];
    const float qlen = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    const float inv_len = qlen > 0.f ? 1.f / qlen : 0.f;
    host[o + 4] = qw * inv_len;
    host[o + 5] = qx * inv_len;
    host[o + 6] = qy * inv_len;
    host[o + 7] = qz * inv_len;
    host[o + 8] = scale_x[i];
    host[o + 9] = scale_y[i];
    host[o + 10] = scale_z[i];
    host[o + 11] = extent_x[i];
    host[o + 12] = extent_y[i];
    host[o + 13] = extent_z[i];
    host[o + 14] = 0.f;
    host[o + 15] = 0.f;
  }

  cuda_check(cudaMemcpyAsync(d_gaussians_, host.data(), need_floats * sizeof(float), cudaMemcpyHostToDevice,
                             static_cast<cudaStream_t>(stream_)),
             "cudaMemcpyAsync gaussians");
  cuda_check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)), "cudaStreamSynchronize upload");
  num_gaussians_ = static_cast<int>(n);
}

MultiBatchResult GpuVoxelization::submitMultiBatch(int slot_index, const std::vector<uint32_t>& concatenated_indices,
                                                   size_t total_indices, const std::vector<BatchSpec>& batches,
                                                   float voxel_resolution, float opacity_cutoff) {
  if (slot_index < 0 || slot_index >= kNumSlots) {
    throw std::out_of_range("GpuVoxelization::submitMultiBatch: invalid slot_index");
  }
  if (!d_gaussians_ || num_gaussians_ <= 0) {
    throw std::runtime_error("GpuVoxelization::submitMultiBatch: call uploadAllGaussians first");
  }
  if (total_indices > concatenated_indices.size()) {
    throw std::invalid_argument("GpuVoxelization::submitMultiBatch: total_indices exceeds buffer");
  }

  MultiBatchResult out;
  out.max_blocks_per_batch = kMaxBlocksPerBatch;

  const int num_batches = static_cast<int>(batches.size());
  if (num_batches == 0) {
    return out;
  }

  cudaSetDevice(cuda_device_index_);
  SlotBuffers& slot = slots_[slot_index];

  const size_t index_bytes = total_indices * sizeof(std::uint32_t);
  if (index_bytes > slot.index_cap) {
    cudaFree(slot.d_indices);
    slot.d_indices = nullptr;
    slot.index_cap = 0;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot.d_indices), index_bytes), "cudaMalloc indices");
    slot.index_cap = index_bytes;
  }

  const size_t results_u32_count = static_cast<size_t>(num_batches) * kMaxBlocksPerBatch * 2u;
  const size_t results_bytes = results_u32_count * sizeof(unsigned int);
  if (results_bytes > slot.results_byte_cap) {
    cudaFree(slot.d_results);
    slot.d_results = nullptr;
    slot.results_byte_cap = 0;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot.d_results), results_bytes), "cudaMalloc results");
    slot.results_byte_cap = results_bytes;
  }

  const size_t batch_bytes = static_cast<size_t>(num_batches) * sizeof(BatchInfoHost);
  if (batch_bytes > slot.batch_info_bytes_cap) {
    cudaFree(slot.d_batch_infos);
    slot.d_batch_infos = nullptr;
    slot.batch_info_bytes_cap = 0;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&slot.d_batch_infos), batch_bytes), "cudaMalloc batch_infos");
    slot.batch_info_bytes_cap = batch_bytes;
  }

  cuda_check(cudaMemsetAsync(slot.d_results, 0, results_bytes, static_cast<cudaStream_t>(stream_)),
             "cudaMemsetAsync results");

  cuda_check(cudaMemcpyAsync(slot.d_indices, concatenated_indices.data(), index_bytes, cudaMemcpyHostToDevice,
                             static_cast<cudaStream_t>(stream_)),
             "cudaMemcpyAsync indices");

  std::vector<BatchInfoHost> batch_host(static_cast<size_t>(num_batches));
  for (int i = 0; i < num_batches; ++i) {
    const BatchSpec& b = batches[static_cast<size_t>(i)];
    batch_host[static_cast<size_t>(i)].index_offset = b.index_offset;
    batch_host[static_cast<size_t>(i)].index_count = b.index_count;
    batch_host[static_cast<size_t>(i)].num_blocks_x = b.num_blocks_x;
    batch_host[static_cast<size_t>(i)].num_blocks_y = b.num_blocks_y;
    batch_host[static_cast<size_t>(i)].num_blocks_z = b.num_blocks_z;
    batch_host[static_cast<size_t>(i)].block_min_x = b.block_min_x;
    batch_host[static_cast<size_t>(i)].block_min_y = b.block_min_y;
    batch_host[static_cast<size_t>(i)].block_min_z = b.block_min_z;
  }

  cuda_check(cudaMemcpyAsync(slot.d_batch_infos, batch_host.data(), batch_bytes, cudaMemcpyHostToDevice,
                             static_cast<cudaStream_t>(stream_)),
             "cudaMemcpyAsync batch_infos");

  cudaStream_t st = static_cast<cudaStream_t>(stream_);
  cudaError_t launch_err = splat_cuda_voxel_multibatch_launch(
      st, opacity_cutoff, voxel_resolution, static_cast<unsigned int>(kMaxBlocksPerBatch), d_gaussians_, slot.d_indices,
      slot.d_results, slot.d_batch_infos, num_batches);
  cuda_check(launch_err, "splat_cuda_voxel_multibatch_launch");

  out.masks.resize(results_u32_count);
  cuda_check(cudaMemcpyAsync(out.masks.data(), slot.d_results, results_bytes, cudaMemcpyDeviceToHost, st),
             "cudaMemcpyAsync masks D2H");
  cuda_check(cudaStreamSynchronize(st), "cudaStreamSynchronize submit");

  return out;
}

}  // namespace splat
