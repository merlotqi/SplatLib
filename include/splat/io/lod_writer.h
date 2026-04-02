/**
 * @file splat/io/lod_writer.h
 * @brief Write level-of-detail (LOD) Gaussian splat data
 *
 * Provides functionality to write Gaussian splat data in LOD format,
 * supporting chunked output and environment map data.
 */
 
#pragma once

#include <splat/models/data-table.h>

namespace splat {

/**
 * @brief Write splat data to LOD format output file
 *
 * Generates a multi-level-of-detail representation of the input splat data,
 * with optional environment map and bundling.
 *
 * @param filename Output file path
 * @param dataTable Primary splat data table
 * @param envDataTable Optional environment map data table (nullptr if not used)
 * @param bundle Whether to bundle data into a single file
 * @param iterations Number of simplification iterations per LOD level
 * @param lodChunkCount Number of splats per LOD chunk
 * @param lodChunkExtent Spatial extent of each LOD chunk
 */
void writeLod(const std::string& filename, const DataTable* dataTable, DataTable* envDataTable, bool bundle,
              int iterations, size_t lodChunkCount, size_t lodChunkExtent);

}  // namespace splat
