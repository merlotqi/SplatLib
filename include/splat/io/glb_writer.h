/**
 * @file splat/io/glb_writer.h
 * @brief Encode splats as GLB with `KHR_gaussian_splatting`.
 *
 * References:
 * - [glTF 2.0](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
 * - [KHR_gaussian_splatting (glTF PR)](https://github.com/KhronosGroup/glTF/pull/2421)
 */
 
#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace splat {

class DataTable;

/**
 * @brief Options for writeGlb (port of splat-transform WriteGlbOptions).
 */
struct WriteGlbOptions {
  std::filesystem::path filename;
  const DataTable* data_table = nullptr;
};

/**
 * @brief Encode a DataTable as GLB with KHR_gaussian_splatting (matches write-glb.ts).
 *
 * Required columns: x,y,z, rot_0..3, scale_0..2, opacity, f_dc_0..2.
 * Optional: f_rest_0..f_rest_44 for higher-order SH (same band detection as TS).
 */
std::vector<uint8_t> buildGaussianSplatGlb(const DataTable& data_table);

/**
 * @brief Write Gaussian splat GLB to disk.
 */
void writeGlb(const std::filesystem::path& filename, const DataTable* data_table);

}  // namespace splat
