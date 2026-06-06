/**
 * @file splat/io/ply_reader.h
 * @brief Read PLY Gaussian splat files.
 *
 * References:
 * - [PLY format](http://paulbourke.net/dataformats/ply/)
 * - [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
 */

#pragma once

#include <filesystem>
#include <memory>

namespace splat {

class SplatCloud;

/**
 * @brief Reads and parses a PLY (Polygon File Format) file from disk.
 *
 * This function loads a PLY file, parses its header and data sections, and returns
 * a SplatCloud containing the vertex data. The function supports both ASCII and binary
 * PLY formats and handles data in chunks for memory efficiency.
 *
 * @param[in] filename Path to the PLY file to be read.
 *
 * @return std::unique_ptr<SplatCloud> A smart pointer to a SplatCloud containing
 *         the vertex data from the PLY file. If the file contains compressed data,
 *         it will be decompressed automatically.
 *
 * @throws std::runtime_error If:
 *         - The file cannot be opened
 *         - The file header is invalid or missing the PLY magic bytes
 *         - The header exceeds the maximum size (128KB)
 *         - The 'end_header' marker is not found
 *         - Data chunks cannot be read properly
 *         - The file does not contain a vertex element
 *
 * @note The function reads data in chunks of 1024 rows at a time to optimize memory usage.
 *       Non-vertex elements are stored in the PlyData structure but only vertex data is returned.
 *       If the PLY data is compressed, it will be decompressed before returning.
 *
 * @par File Format Support:
 * - Binary PLY format (little-endian)
 * - Structured elements with properties
 * - Optional compression (automatically detected and decompressed)
 *
 * @see parseHeader() For header parsing details
 * @see isCompressedPly() For compression detection
 * @see decompressPly() For decompression logic
 */
std::unique_ptr<SplatCloud> readPly(const std::filesystem::path& filename);

}  // namespace splat
