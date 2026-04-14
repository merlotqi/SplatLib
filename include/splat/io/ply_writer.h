/**
 * @file splat/io/ply_writer.h
 * @brief Write PLY Gaussian splat files.
 *
 * References:
 * - [PLY format](http://paulbourke.net/dataformats/ply/)
 * - [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
 */
 
#pragma once

#include <splat/models/ply.h>
#include <filesystem>

namespace splat {

/**
 * @brief Write Gaussian splat data to a PLY file
 *
 * Serializes splat data (positions, colors, covariances, SH coefficients)
 * into a binary PLY file format compatible with 3D Gaussian Splatting tools.
 *
 * @param filename Output file path
 * @param plyData PLY data structure containing splat attributes
 */
void writePly(const std::filesystem::path& filename, const PlyData& plyData);

}  // namespace splat
