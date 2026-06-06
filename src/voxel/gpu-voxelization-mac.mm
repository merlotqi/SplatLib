#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <splat/models/splatcloud.h>
#include <splat/voxel/gpu-voxelization.h>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
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

struct KernelParams {
  float opacity_cutoff;
  float voxel_resolution;
  std::uint32_t max_blocks_per_batch;
  std::uint32_t _pad;
};

static_assert(sizeof(BatchInfoHost) == 32, "BatchInfoHost layout");
static_assert(sizeof(KernelParams) == 16, "KernelParams layout");

struct MetalVoxelContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLBuffer> gaussians_buffer = nil;
  size_t gaussians_bytes = 0;

  ~MetalVoxelContext() {
    if (gaussians_buffer) {
      [gaussians_buffer release];
      gaussians_buffer = nil;
    }
    if (pipeline) {
      [pipeline release];
      pipeline = nil;
    }
    if (library) {
      [library release];
      library = nil;
    }
    if (queue) {
      [queue release];
      queue = nil;
    }
    if (device) {
      [device release];
      device = nil;
    }
  }
};

static std::string metal_shader_path() {
  std::string self = __FILE__;
  const size_t slash = self.find_last_of('/');
  if (slash == std::string::npos) {
    return "gpu-voxelization-metal.metal";
  }
  return self.substr(0, slash + 1) + "gpu-voxelization-metal.metal";
}

static std::string load_file_text(const std::string& path) {
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) {
    throw std::runtime_error("GpuVoxelization(mac): cannot open Metal shader: " + path);
  }
  file.seekg(0, std::ios::end);
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::string text;
  text.resize(static_cast<size_t>(size));
  if (size > 0) {
    file.read(&text[0], size);
  }
  return text;
}

static MetalVoxelContext* context_from_stream(void* stream) {
  return static_cast<MetalVoxelContext*>(stream);
}

static const MetalVoxelContext* context_from_stream(const void* stream) {
  return static_cast<const MetalVoxelContext*>(stream);
}

}  // namespace

bool gpuVoxelizationIsAvailable() noexcept {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device) {
    [device release];
    return true;
  }
  return false;
}

GpuVoxelization::GpuVoxelization(int cuda_device_index) : cuda_device_index_(cuda_device_index) {
  if (cuda_device_index_ < 0) {
    throw std::runtime_error("GpuVoxelization(mac): invalid device index");
  }

  MetalVoxelContext* ctx = new MetalVoxelContext();
  ctx->device = MTLCreateSystemDefaultDevice();
  if (!ctx->device) {
    delete ctx;
    throw std::runtime_error("GpuVoxelization(mac): no Metal device");
  }

  ctx->queue = [ctx->device newCommandQueue];
  if (!ctx->queue) {
    delete ctx;
    throw std::runtime_error("GpuVoxelization(mac): failed to create command queue");
  }

  NSError* error = nil;
  const std::string shader_text = load_file_text(metal_shader_path());
  NSString* source = [NSString stringWithUTF8String:shader_text.c_str()];
  if (!source) {
    delete ctx;
    throw std::runtime_error("GpuVoxelization(mac): failed to prepare Metal shader source");
  }

  MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
  options.fastMathEnabled = YES;
  ctx->library = [ctx->device newLibraryWithSource:source options:options error:&error];
  [options release];
  if (!ctx->library) {
    const char* err = error ? [[error localizedDescription] UTF8String] : "unknown";
    delete ctx;
    throw std::runtime_error(std::string("GpuVoxelization(mac): Metal library compile failed: ") + err);
  }

  id<MTLFunction> fn = [ctx->library newFunctionWithName:@"voxelize_multibatch_kernel"];
  if (!fn) {
    delete ctx;
    throw std::runtime_error("GpuVoxelization(mac): missing kernel voxelize_multibatch_kernel");
  }

  ctx->pipeline = [ctx->device newComputePipelineStateWithFunction:fn error:&error];
  [fn release];
  if (!ctx->pipeline) {
    const char* err = error ? [[error localizedDescription] UTF8String] : "unknown";
    delete ctx;
    throw std::runtime_error(std::string("GpuVoxelization(mac): failed to create pipeline: ") + err);
  }

  stream_ = ctx;
}

void GpuVoxelization::freeSlot(SlotBuffers& s) {
  s.d_indices = nullptr;
  s.d_results = nullptr;
  s.d_batch_infos = nullptr;
  s.index_cap = 0;
  s.results_byte_cap = 0;
  s.batch_info_bytes_cap = 0;
}

void GpuVoxelization::destroyStream() {
  if (stream_) {
    MetalVoxelContext* ctx = context_from_stream(stream_);
    delete ctx;
    stream_ = nullptr;
  }
}

GpuVoxelization::~GpuVoxelization() {
  freeSlot(slots_[0]);
  freeSlot(slots_[1]);
  delete[] d_gaussians_;
  d_gaussians_ = nullptr;
  gaussian_buffer_floats_ = 0;
  destroyStream();
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
    freeSlot(slots_[0]);
    freeSlot(slots_[1]);
    delete[] d_gaussians_;
    destroyStream();

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
    delete[] d_gaussians_;
    d_gaussians_ = new float[need_floats];
    gaussian_buffer_floats_ = need_floats;
  }

  for (size_t i = 0; i < n; ++i) {
    const size_t o = i * static_cast<size_t>(kFloatsPerGaussian);
    d_gaussians_[o + 0] = x[i];
    d_gaussians_[o + 1] = y[i];
    d_gaussians_[o + 2] = z[i];
    d_gaussians_[o + 3] = opacity[i];
    float qw = rot_w[i], qx = rot_x[i], qy = rot_y[i], qz = rot_z[i];
    const float qlen = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    const float inv_len = qlen > 0.f ? 1.f / qlen : 0.f;
    d_gaussians_[o + 4] = qw * inv_len;
    d_gaussians_[o + 5] = qx * inv_len;
    d_gaussians_[o + 6] = qy * inv_len;
    d_gaussians_[o + 7] = qz * inv_len;
    d_gaussians_[o + 8] = scale_x[i];
    d_gaussians_[o + 9] = scale_y[i];
    d_gaussians_[o + 10] = scale_z[i];
    d_gaussians_[o + 11] = extent_x[i];
    d_gaussians_[o + 12] = extent_y[i];
    d_gaussians_[o + 13] = extent_z[i];
    d_gaussians_[o + 14] = 0.f;
    d_gaussians_[o + 15] = 0.f;
  }

  MetalVoxelContext* ctx = context_from_stream(stream_);
  const size_t bytes = need_floats * sizeof(float);
  if (ctx->gaussians_buffer) {
    [ctx->gaussians_buffer release];
    ctx->gaussians_buffer = nil;
  }
  ctx->gaussians_buffer = [ctx->device newBufferWithBytes:d_gaussians_ length:bytes options:MTLResourceStorageModeShared];
  if (!ctx->gaussians_buffer) {
    throw std::runtime_error("GpuVoxelization(mac): failed to create gaussians buffer");
  }
  ctx->gaussians_bytes = bytes;
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

  MetalVoxelContext* ctx = context_from_stream(stream_);
  if (!ctx || !ctx->gaussians_buffer) {
    throw std::runtime_error("GpuVoxelization(mac): gaussians Metal buffer is not initialized");
  }

  const size_t index_bytes = total_indices * sizeof(std::uint32_t);
  const size_t batch_bytes = static_cast<size_t>(num_batches) * sizeof(BatchInfoHost);
  const size_t results_u32_count = static_cast<size_t>(num_batches) * kMaxBlocksPerBatch * 2u;
  const size_t results_bytes = results_u32_count * sizeof(std::uint32_t);

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

  id<MTLBuffer> indices_buffer = [ctx->device newBufferWithBytes:concatenated_indices.data()
                                                           length:index_bytes
                                                          options:MTLResourceStorageModeShared];
  if (!indices_buffer) {
    throw std::runtime_error("GpuVoxelization(mac): failed to create indices buffer");
  }

  id<MTLBuffer> batch_buffer = [ctx->device newBufferWithBytes:batch_host.data()
                                                         length:batch_bytes
                                                        options:MTLResourceStorageModeShared];
  if (!batch_buffer) {
    [indices_buffer release];
    throw std::runtime_error("GpuVoxelization(mac): failed to create batch buffer");
  }

  id<MTLBuffer> results_buffer = [ctx->device newBufferWithLength:results_bytes options:MTLResourceStorageModeShared];
  if (!results_buffer) {
    [batch_buffer release];
    [indices_buffer release];
    throw std::runtime_error("GpuVoxelization(mac): failed to create results buffer");
  }
  std::memset([results_buffer contents], 0, results_bytes);

  KernelParams params{};
  params.opacity_cutoff = opacity_cutoff;
  params.voxel_resolution = voxel_resolution;
  params.max_blocks_per_batch = kMaxBlocksPerBatch;

  id<MTLCommandBuffer> command_buffer = [ctx->queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  [encoder setComputePipelineState:ctx->pipeline];
  [encoder setBuffer:ctx->gaussians_buffer offset:0 atIndex:0];
  [encoder setBuffer:indices_buffer offset:0 atIndex:1];
  [encoder setBuffer:results_buffer offset:0 atIndex:2];
  [encoder setBuffer:batch_buffer offset:0 atIndex:3];
  [encoder setBytes:&params length:sizeof(params) atIndex:4];

  const MTLSize tg_size = MTLSizeMake(64, 1, 1);
  const MTLSize tg_count = MTLSizeMake(static_cast<NSUInteger>(kMaxBlocksPerBatch), static_cast<NSUInteger>(num_batches), 1);
  [encoder dispatchThreadgroups:tg_count threadsPerThreadgroup:tg_size];
  [encoder endEncoding];

  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  if (command_buffer.status == MTLCommandBufferStatusError) {
    const char* err = command_buffer.error ? [[command_buffer.error localizedDescription] UTF8String] : "unknown";
    [results_buffer release];
    [batch_buffer release];
    [indices_buffer release];
    throw std::runtime_error(std::string("GpuVoxelization(mac): command buffer failed: ") + err);
  }

  out.masks.resize(results_u32_count);
  std::memcpy(out.masks.data(), [results_buffer contents], results_bytes);

  [results_buffer release];
  [batch_buffer release];
  [indices_buffer release];

  return out;
}

}  // namespace splat
