/**
 * @file splat/io/lcc_reader.h
 * @brief Read LCC (Layered Compressed Cloud) format data
 *
 * Provides functions to load LCC files into DataTable format for
 * Gaussian splat processing.
 */
 
#pragma once

#include <filesystem>
#include <splat/models/data-table.h>

namespace splat {

/**
 * @brief Load LCC file into DataTable(s)
 *
 * Reads an LCC file and converts the splat data into one or more
 * DataTable objects with typed columns.
 *
 * @param filename Path to the LCC file
 * @param sourceName Source identifier within the LCC file
 * @param options LOD level options (e.g., target LOD indices)
 * @return Vector of DataTable objects containing the loaded splat data
 */
std::vector<std::unique_ptr<DataTable>> readLcc(const std::filesystem::path& filename,
                                                const std::filesystem::path& sourceName,
                                                const std::vector<int>& options);

}  // namespace splat
