/**
 * @file splat/io/voxel_reader.h
 * @brief Read voxel tree JSON/binary sidecar format
 *
 * Provides functions to load voxel-based Gaussian splat data stored
 * as paired JSON metadata and binary data files.
 */
 
#pragma once

#include <splat/models/data-table.h>
#include <filesystem>

namespace splat {

/**
 * Read a .voxel.json file and convert to DataTable (finest/leaf LOD).
 * @brief Load voxel tree data from JSON and binary files
 *
 * Reads a .voxel.json file and its sibling .voxel.bin, expanding
 * leaf voxel blocks into a Gaussian DataTable. Output columns include:
 * x, y, z (position), scale_0..2, rot_0..3, f_dc_0..2 (SH color), opacity.
 *
 * @param voxel_json_path Path to the .voxel.json metadata file
 * @return DataTable containing expanded Gaussian splat data
 */
std::unique_ptr<DataTable> readVoxel(const std::filesystem::path& filename);

}  // namespace splat
