/**
 * @file splat/io/voxel_writer.h
 * @brief Write voxel tree and optional collision mesh GLB.
 *
 * - [meshoptimizer](https://github.com/zeux/meshoptimizer)
 */
 
#pragma once

#include <splat/voxel/nav-simplify.h>

#include <filesystem>
#include <optional>

namespace splat {

class DataTable;

/**
 * @brief Options for writeVoxel (port of splat-transform WriteVoxelOptions).
 *
 * @p mesh_simplify is the fraction of indices to keep in (0, 1], applied via meshoptimizer
 * @c meshopt_simplify with @c meshopt_SimplifyErrorAbsolute and absolute error @p voxel_resolution
 * (matches TS MeshoptSimplifier.simplify + ErrorAbsolute).
 */
struct WriteVoxelOptions {
  std::filesystem::path filename;  ///< Must end with .voxel.json
  const DataTable* data_table = nullptr;
  float voxel_resolution = 0.05f;
  float opacity_cutoff = 0.5f;
  int cuda_device_index = 0;
  bool collision_mesh = false;
  float mesh_simplify = 0.25f;  ///< Target index count ratio (0-1), default 0.25.
  struct NavCapsule {
    float height = 0.f;
    float radius = 0.f;
  };
  std::optional<NavCapsule> nav_capsule;
  std::optional<NavSeed> nav_seed;
};

/**
 * @brief GPU voxelize Gaussians and write .voxel.json + .voxel.bin (+ optional .collision.glb).
 *
 * Requires a CUDA device. Matches splat-transform write-voxel.ts ordering; GPU submits are
 * sequential (fully synchronized) but produce the same masks as the TS pipeline.
 */
void writeVoxel(const WriteVoxelOptions& options);

}  // namespace splat
