/**
 * @file splat/io/compressed_ply_writer.h
 * @brief Write compressed PLY-style streams.
 *
 * References:
 * - [PLY format](http://paulbourke.net/dataformats/ply/)
 */
 
#pragma once

#include <filesystem>
#include <splat/models/data-table.h>

namespace splat {

/**
 * @brief Write Gaussian splat data to compressed PLY format
 *
 * Encodes splat data using quantized and bit-packed compression
 * for efficient storage. The output is compatible with compressed
 * PLY readers.
 *
 * @param filename Output file path
 * @param dataTable Splat data table to encode
 */
void writeCompressedPly(const std::filesystem::path& filename, DataTable* dataTable);

}  // namespace splat
