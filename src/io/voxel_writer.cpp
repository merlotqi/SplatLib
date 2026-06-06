#include <absl/types/span.h>
#include <meshoptimizer.h>
#include <splat/io/voxel_writer.h>
#include <splat/models/splatcloud.h>
#include <splat/op/morton_order.h>
#include <splat/op/transform.h>
#include <splat/utils/logger.h>
#include <splat/voxel/collision-glb.h>
#include <splat/voxel/gaussian-aabb.h>
#include <splat/voxel/gaussian-bvh.h>
#include <splat/voxel/gpu-voxelization.h>
#include <splat/voxel/marching-cubes.h>
#include <splat/voxel/nav-simplify.h>
#include <splat/voxel/sparse-octree.h>
#include <splat/voxel/voxel-filter.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace splat {

namespace {

constexpr int kMegaMaxBatches = 512;
constexpr size_t kMegaMaxIndices = 4u * 1024u * 1024u;
constexpr int kBatchSize = 16;
constexpr float kPi = 3.14159265358979323846f;

struct PendingBatch {
  uint32_t index_offset = 0;
  uint32_t index_count = 0;
  float block_min_x = 0.f;
  float block_min_y = 0.f;
  float block_min_z = 0.f;
  uint32_t num_blocks_x = 0;
  uint32_t num_blocks_y = 0;
  uint32_t num_blocks_z = 0;
  int bx = 0;
  int by = 0;
  int bz = 0;
};

static void processResults(const std::vector<uint32_t>& masks, const std::vector<PendingBatch>& batches,
                           uint32_t max_blocks_per_batch, BlockAccumulator& accumulator) {
  for (size_t b = 0; b < batches.size(); ++b) {
    const PendingBatch& batch = batches[b];
    const size_t batch_result_offset = b * static_cast<size_t>(max_blocks_per_batch) * 2u;
    const uint32_t total_batch_blocks = batch.num_blocks_x * batch.num_blocks_y * batch.num_blocks_z;
    for (uint32_t block_idx = 0; block_idx < total_batch_blocks; ++block_idx) {
      uint32_t mask_lo = masks[batch_result_offset + static_cast<size_t>(block_idx) * 2u];
      uint32_t mask_hi = masks[batch_result_offset + static_cast<size_t>(block_idx) * 2u + 1u];
      if (mask_lo == 0 && mask_hi == 0) {
        continue;
      }
      uint32_t local_x = block_idx % batch.num_blocks_x;
      uint32_t local_y = (block_idx / batch.num_blocks_x) % batch.num_blocks_y;
      uint32_t local_z = block_idx / (batch.num_blocks_x * batch.num_blocks_y);
      const int abs_bx = batch.bx + static_cast<int>(local_x);
      const int abs_by = batch.by + static_cast<int>(local_y);
      const int abs_bz = batch.bz + static_cast<int>(local_z);
      uint32_t morton =
          xyzToMorton(static_cast<uint32_t>(abs_bx), static_cast<uint32_t>(abs_by), static_cast<uint32_t>(abs_bz));
      accumulator.addBlock(morton, mask_lo, mask_hi);
    }
  }
}

static void flushPendingBatches(GpuVoxelization& gpu, int slot_index, std::vector<PendingBatch>& pending,
                                std::vector<uint32_t>& slot_indices, size_t& index_offset, float voxel_resolution,
                                float opacity_cutoff, BlockAccumulator& accumulator) {
  if (pending.empty()) {
    return;
  }
  std::vector<BatchSpec> specs;
  specs.reserve(pending.size());
  for (const PendingBatch& p : pending) {
    BatchSpec b;
    b.index_offset = p.index_offset;
    b.index_count = p.index_count;
    b.block_min_x = p.block_min_x;
    b.block_min_y = p.block_min_y;
    b.block_min_z = p.block_min_z;
    b.num_blocks_x = p.num_blocks_x;
    b.num_blocks_y = p.num_blocks_y;
    b.num_blocks_z = p.num_blocks_z;
    specs.push_back(b);
  }
  std::vector<uint32_t> concat(slot_indices.begin(), slot_indices.begin() + static_cast<std::ptrdiff_t>(index_offset));
  MultiBatchResult r = gpu.submitMultiBatch(slot_index, concat, index_offset, specs, voxel_resolution, opacity_cutoff);
  processResults(r.masks, pending, r.max_blocks_per_batch, accumulator);
  pending.clear();
  index_offset = 0;
}

static void validateVoxelJsonPath(const std::filesystem::path& p) {
  std::string s = p.filename().string();
  static const char kSuffix[] = ".voxel.json";
  constexpr size_t kLen = sizeof(kSuffix) - 1;
  if (s.size() < kLen) {
    throw std::invalid_argument("writeVoxel: filename must end with .voxel.json");
  }
  bool ok = true;
  for (size_t i = 0; i < kLen; ++i) {
    if (std::tolower(static_cast<unsigned char>(s[s.size() - kLen + i])) != static_cast<unsigned char>(kSuffix[i])) {
      ok = false;
      break;
    }
  }
  if (!ok) {
    throw std::invalid_argument("writeVoxel: filename must end with .voxel.json");
  }
}

static std::filesystem::path voxelBinPath(const std::filesystem::path& json_path) {
  std::string s = json_path.filename().string();
  static const char kFrom[] = ".voxel.json";
  constexpr size_t kLen = sizeof(kFrom) - 1;
  s.replace(s.size() - kLen, kLen, ".voxel.bin");
  return json_path.parent_path() / s;
}

static std::filesystem::path collisionGlbPath(const std::filesystem::path& json_path) {
  std::string s = json_path.filename().string();
  static const char kFrom[] = ".voxel.json";
  constexpr size_t kLen = sizeof(kFrom) - 1;
  s.replace(s.size() - kLen, kLen, ".collision.glb");
  return json_path.parent_path() / s;
}

static void writeOctreeFiles(const std::filesystem::path& json_path, const SparseOctree& octree) {
  nlohmann::json meta;
  meta["version"] = "1.1";
  meta["gridBounds"]["min"] = {octree.gridBounds.min.x(), octree.gridBounds.min.y(), octree.gridBounds.min.z()};
  meta["gridBounds"]["max"] = {octree.gridBounds.max.x(), octree.gridBounds.max.y(), octree.gridBounds.max.z()};
  meta["sceneBounds"]["min"] = {octree.sceneBounds.min.x(), octree.sceneBounds.min.y(), octree.sceneBounds.min.z()};
  meta["sceneBounds"]["max"] = {octree.sceneBounds.max.x(), octree.sceneBounds.max.y(), octree.sceneBounds.max.z()};
  meta["voxelResolution"] = octree.voxelResolution;
  meta["leafSize"] = octree.leafSize;
  meta["treeDepth"] = octree.treeDepth;
  meta["numInteriorNodes"] = octree.numInteriorNodes;
  meta["numMixedLeaves"] = octree.numMixedLeaves;
  meta["nodeCount"] = octree.nodes.size();
  meta["leafDataCount"] = octree.leafData.size();

  {
    std::ofstream jf(json_path);
    if (!jf) {
      throw std::runtime_error("writeVoxel: cannot open for write: " + json_path.string());
    }
    jf << meta.dump(2);
  }

  const std::filesystem::path bin_path = voxelBinPath(json_path);
  std::ofstream bf(bin_path, std::ios::binary);
  if (!bf) {
    throw std::runtime_error("writeVoxel: cannot open bin for write: " + bin_path.string());
  }
  if (!octree.nodes.empty()) {
    bf.write(reinterpret_cast<const char*>(octree.nodes.data()),
             static_cast<std::streamsize>(octree.nodes.size() * sizeof(uint32_t)));
  }
  if (!octree.leafData.empty()) {
    bf.write(reinterpret_cast<const char*>(octree.leafData.data()),
             static_cast<std::streamsize>(octree.leafData.size() * sizeof(uint32_t)));
  }
}

/**
 * Port of write-voxel.ts collision simplification: meshopt_simplify with ErrorAbsolute,
 * then compact vertices (remap) like TS MeshoptSimplifier output handling.
 *
 * @return true if @p mesh was replaced with simplified+compact data; false if GLB should be skipped
 *         (matches TS: no fallback to raw marching-cubes mesh for output).
 */
static bool simplifyCollisionMesh(MarchingCubesMesh& mesh, float voxel_resolution, float mesh_simplify_ratio) {
  if (mesh.indices.size() < 3) {
    return false;
  }
  const size_t vertex_count = mesh.positions.size() / 3;
  if (vertex_count == 0) {
    return false;
  }

  const float clamped = std::isfinite(mesh_simplify_ratio) ? std::min(1.f, std::max(0.f, mesh_simplify_ratio)) : 0.25f;
  const size_t target_index_count = std::max(
      size_t{3}, std::min(mesh.indices.size(), static_cast<size_t>(std::floor(static_cast<double>(mesh.indices.size()) *
                                                                              static_cast<double>(clamped) / 3.0)) *
                                                   3u));

  std::vector<unsigned int> simplified(mesh.indices.size());
  const size_t nidx =
      meshopt_simplify(simplified.data(), mesh.indices.data(), mesh.indices.size(), mesh.positions.data(), vertex_count,
                       sizeof(float) * 3, target_index_count, voxel_resolution,
                       static_cast<unsigned int>(meshopt_SimplifyErrorAbsolute), nullptr);

  if (nidx < 3) {
    LOG_WARN(
        "writeVoxel: collision mesh: simplification failed or empty (meshopt_simplify returned %zu indices); "
        "skipping GLB output",
        nidx);
    return false;
  }
  simplified.resize(nidx);

  std::unordered_map<unsigned int, unsigned int> vertex_remap;
  vertex_remap.reserve(simplified.size());
  unsigned int new_vertex_count = 0;
  for (unsigned int idx : simplified) {
    if (vertex_remap.find(idx) == vertex_remap.end()) {
      vertex_remap.emplace(idx, new_vertex_count++);
    }
  }

  std::vector<float> compact_positions(static_cast<size_t>(new_vertex_count) * 3u);
  for (const auto& kv : vertex_remap) {
    const unsigned int old_idx = kv.first;
    const unsigned int nw = kv.second;
    compact_positions[nw * 3 + 0] = mesh.positions[old_idx * 3 + 0];
    compact_positions[nw * 3 + 1] = mesh.positions[old_idx * 3 + 1];
    compact_positions[nw * 3 + 2] = mesh.positions[old_idx * 3 + 2];
  }

  std::vector<uint32_t> compact_indices(simplified.size());
  for (size_t i = 0; i < simplified.size(); ++i) {
    compact_indices[i] = vertex_remap[simplified[i]];
  }

  const double reduction =
      (1.0 - static_cast<double>(simplified.size()) / static_cast<double>(mesh.indices.size())) * 100.0;
  LOG_INFO("writeVoxel: collision mesh simplified: %u vertices, %zu triangles (%.0f%% index reduction)",
           new_vertex_count, simplified.size() / 3, reduction);

  mesh.positions = std::move(compact_positions);
  mesh.indices = std::move(compact_indices);
  return true;
}

}  // namespace

void writeVoxel(const WriteVoxelOptions& options) {
  if (!options.data_table) {
    throw std::invalid_argument("writeVoxel: data_table is required");
  }
  if (!gpuVoxelizationIsAvailable()) {
    throw std::runtime_error("writeVoxel: CUDA is required for GPU voxelization");
  }
  validateVoxelJsonPath(options.filename);

  const bool nav_caps = options.nav_capsule.has_value();
  const bool nav_seed = options.nav_seed.has_value();
  if (nav_caps != nav_seed) {
    LOG_WARN("writeVoxel: both nav_capsule and nav_seed must be provided for nav simplification; skipping");
  }
  const bool has_nav = nav_caps && nav_seed;

  static const std::vector<std::string> kVoxelColumns = {
      "x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2", "opacity",
  };
  std::unique_ptr<SplatCloud> pc_table = options.data_table->clone(kVoxelColumns);

  Eigen::Quaternionf q(Eigen::AngleAxisf(kPi, Eigen::Vector3f::UnitZ()));
  transform(pc_table.get(), Eigen::Vector3f::Zero(), q, 1.f);

  GaussianExtentsResult extents_result = computeGaussianExtents(*pc_table);
  Bounds scene_bounds;
  scene_bounds.min = extents_result.sceneMin;
  scene_bounds.max = extents_result.sceneMax;

  GaussianBVH bvh(*pc_table, *extents_result.extents);
  GpuVoxelization gpu(options.cuda_device_index);
  gpu.uploadAllGaussians(*pc_table, *extents_result.extents);

  const float block_size = 4.f * options.voxel_resolution;
  Bounds grid_bounds =
      alignGridBounds(scene_bounds.min.x(), scene_bounds.min.y(), scene_bounds.min.z(), scene_bounds.max.x(),
                      scene_bounds.max.y(), scene_bounds.max.z(), options.voxel_resolution);

  const int num_blocks_x = static_cast<int>(std::lround((grid_bounds.max.x() - grid_bounds.min.x()) / block_size));
  const int num_blocks_y = static_cast<int>(std::lround((grid_bounds.max.y() - grid_bounds.min.y()) / block_size));
  const int num_blocks_z = static_cast<int>(std::lround((grid_bounds.max.z() - grid_bounds.min.z()) / block_size));

  BlockAccumulator accumulator;
  std::vector<PendingBatch> pending;
  pending.reserve(256);

  size_t slot_capacity = 1024u * 1024u;
  std::vector<uint32_t> slot_indices(slot_capacity);
  int current_slot = 0;
  size_t index_offset = 0;

  for (int bz = 0; bz < num_blocks_z; bz += kBatchSize) {
    for (int by = 0; by < num_blocks_y; by += kBatchSize) {
      for (int bx = 0; bx < num_blocks_x; bx += kBatchSize) {
        const int curr_batch_x = std::min(kBatchSize, num_blocks_x - bx);
        const int curr_batch_y = std::min(kBatchSize, num_blocks_y - by);
        const int curr_batch_z = std::min(kBatchSize, num_blocks_z - bz);

        const float block_min_x = grid_bounds.min.x() + static_cast<float>(bx) * block_size;
        const float block_min_y = grid_bounds.min.y() + static_cast<float>(by) * block_size;
        const float block_min_z = grid_bounds.min.z() + static_cast<float>(bz) * block_size;
        const float block_max_x = block_min_x + static_cast<float>(curr_batch_x) * block_size;
        const float block_max_y = block_min_y + static_cast<float>(curr_batch_y) * block_size;
        const float block_max_z = block_min_z + static_cast<float>(curr_batch_z) * block_size;

        std::vector<uint32_t> overlapping =
            bvh.queryOverlappingRaw(block_min_x, block_min_y, block_min_z, block_max_x, block_max_y, block_max_z);
        if (overlapping.empty()) {
          continue;
        }

        const size_t needed = index_offset + overlapping.size();
        if (needed > slot_capacity) {
          slot_capacity = std::max(slot_capacity * 2, needed);
          slot_indices.resize(slot_capacity);
        }
        std::copy(overlapping.begin(), overlapping.end(),
                  slot_indices.begin() + static_cast<std::ptrdiff_t>(index_offset));

        PendingBatch pb;
        pb.index_offset = static_cast<uint32_t>(index_offset);
        pb.index_count = static_cast<uint32_t>(overlapping.size());
        pb.block_min_x = block_min_x;
        pb.block_min_y = block_min_y;
        pb.block_min_z = block_min_z;
        pb.num_blocks_x = static_cast<uint32_t>(curr_batch_x);
        pb.num_blocks_y = static_cast<uint32_t>(curr_batch_y);
        pb.num_blocks_z = static_cast<uint32_t>(curr_batch_z);
        pb.bx = bx;
        pb.by = by;
        pb.bz = bz;
        pending.push_back(pb);
        index_offset += overlapping.size();

        if (pending.size() >= static_cast<size_t>(kMegaMaxBatches) || index_offset >= kMegaMaxIndices) {
          flushPendingBatches(gpu, current_slot, pending, slot_indices, index_offset, options.voxel_resolution,
                              options.opacity_cutoff, accumulator);
          current_slot = (current_slot + 1) % GpuVoxelization::kNumSlots;
        }
      }
    }
  }

  flushPendingBatches(gpu, current_slot, pending, slot_indices, index_offset, options.voxel_resolution,
                      options.opacity_cutoff, accumulator);

  accumulator = filterAndFillBlocks(accumulator);

  if (has_nav) {
    NavSimplifyResult nav_result =
        simplifyForCapsule(accumulator, grid_bounds, options.voxel_resolution, options.nav_capsule->height,
                           options.nav_capsule->radius, *options.nav_seed);
    accumulator = std::move(*nav_result.accumulator);
    grid_bounds = nav_result.gridBounds;
  }

  std::vector<uint8_t> glb_bytes;
  if (options.collision_mesh) {
    MarchingCubesMesh collision_mesh = marchingCubes(accumulator, grid_bounds, options.voxel_resolution);
    if (collision_mesh.indices.size() >= 3) {
      LOG_INFO("writeVoxel: collision mesh (raw): %zu vertices, %zu triangles", collision_mesh.positions.size() / 3,
               collision_mesh.indices.size() / 3);
      if (simplifyCollisionMesh(collision_mesh, options.voxel_resolution, options.mesh_simplify)) {
        glb_bytes = buildCollisionGlb(absl::MakeSpan(collision_mesh.positions), absl::MakeSpan(collision_mesh.indices));
      }
    }
  }

  SparseOctree octree = buildSparseOctree(accumulator, grid_bounds, scene_bounds, options.voxel_resolution);

  writeOctreeFiles(options.filename, octree);

  if (!glb_bytes.empty()) {
    const std::filesystem::path glb_path = collisionGlbPath(options.filename);
    std::ofstream gf(glb_path, std::ios::binary);
    if (!gf) {
      throw std::runtime_error("writeVoxel: cannot open collision glb: " + glb_path.string());
    }
    gf.write(reinterpret_cast<const char*>(glb_bytes.data()), static_cast<std::streamsize>(glb_bytes.size()));
  }

  LOG_INFO("writeVoxel: wrote %s (nodes=%llu leafData=%llu)", options.filename.string().c_str(),
           static_cast<unsigned long long>(octree.nodes.size()),
           static_cast<unsigned long long>(octree.leafData.size()));
}

}  // namespace splat
