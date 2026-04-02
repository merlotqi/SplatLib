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
