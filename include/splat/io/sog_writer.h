/**
 * @file splat/io/sog_writer.h
 * @brief Write SOG (Splat on Grid) compressed format
 *
 * Provides functionality to encode Gaussian splat data into the SOG format,
 * which uses texture-based compression for efficient storage.
 */
 
#pragma once

#include <filesystem>
#include <splat/models/data-table.h>

namespace splat {

/**
 * @brief Write Gaussian splat data to SOG format
 *
 * Encodes splat data into compressed SOG format with optional bundling
 * and K-means optimization iterations.
 *
 * @param filename Output file path
 * @param dataTable Splat data table to encode
 * @param bundle Whether to bundle output into a single ZIP file
 * @param iterations K-means optimization iterations for codebook generation
 * @param indices Optional subset of row indices to write (empty = all rows)
 */
void writeSog(const std::filesystem::path& filename, const DataTable* dataTable, bool bundle, int iterations,
              const std::vector<uint32_t>& indices = {});

}  // namespace splat
