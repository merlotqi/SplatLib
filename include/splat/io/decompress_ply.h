/**
 * @file splat/io/decompress_ply.h
 * @brief Decompress PLY payload to `DataTable`.
 *
 * References:
 * - [PLY format](http://paulbourke.net/dataformats/ply/)
 */
 
#pragma once

#include <splat/models/ply.h>

namespace splat {

/**
 * @brief Check if PLY data uses compressed encoding
 *
 * Detects whether the PLY data contains compressed splat attributes
 * that need decompression before use.
 *
 * @param ply PLY data to check
 * @return true if the data is compressed
 */
bool isCompressedPly(const PlyData* ply);

/**
 * @brief Decompress PLY data to DataTable
 *
 * Converts compressed PLY splat attributes back to standard float columns
 * suitable for processing.
 *
 * @param ply Input compressed PLY data
 * @return DataTable with decompressed Gaussian splat data
 */
std::unique_ptr<DataTable> decompressPly(const PlyData* ply);

}  // namespace splat
