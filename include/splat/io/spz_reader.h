/**
 * @file splat/io/spz_reader.h
 * @brief Read `.spz` compressed splats.
 *
 * References:
 * - [.spz format](https://scaniverse.com/news/spz-open-source-gaussian-splat-file-format)
 */
 
#pragma once

#include <filesystem>
#include <splat/models/data-table.h>

namespace splat {

/**
 * @brief Read compressed splat data from .spz format
 *
 * Loads a .spz file (Scaniverse's compressed Gaussian splat format)
 * and decodes it into a DataTable.
 *
 * @param filename Path to the .spz file
 * @return DataTable containing the decoded splat data
 */
std::unique_ptr<DataTable> readSpz(const std::filesystem::path& filename);

}  // namespace splat
