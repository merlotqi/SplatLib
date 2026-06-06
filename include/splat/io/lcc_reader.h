/**
 * @file splat/io/lcc_reader.h
 * @brief Read LCC (Layered Compressed Cloud) format data
 *
 * Provides functions to load LCC files into SplatCloud format for
 * Gaussian splat processing.
 */

#pragma once

#include <splat/models/splatcloud.h>

#include <filesystem>

namespace splat {

/**
 * @brief Load LCC file into SplatCloud(s)
 *
 * Reads an LCC file and converts the splat data into one or more
 * SplatCloud objects with typed columns.
 *
 * @param filename Path to the LCC file
 * @param sourceName Source identifier within the LCC file
 * @param options LOD level options (e.g., target LOD indices)
 * @return Vector of SplatCloud objects containing the loaded splat data
 */
std::vector<std::unique_ptr<SplatCloud>> readLcc(const std::filesystem::path& filename,
                                                 const std::filesystem::path& sourceName,
                                                 const std::vector<int>& options);

}  // namespace splat
