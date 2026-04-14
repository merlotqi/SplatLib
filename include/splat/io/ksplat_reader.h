/**
 * @file splat/io/ksplat_reader.h
 * @brief Read .ksplat container format into DataTable
 *
 * Provides functions to load Kiri Engine's compressed Gaussian splat format.
 */
 
#pragma once

#include <filesystem>
#include <splat/models/data-table.h>

namespace splat {

/**
 * @brief Load Gaussian splat data from .ksplat file
 *
 * Reads a .ksplat file (Kiri Engine's compressed format) and decodes
 * it into a DataTable with standard Gaussian splat columns.
 *
 * @param filename Path to the .ksplat file
 * @return DataTable containing decoded splat data
 */
std::unique_ptr<DataTable> readKsplat(const std::filesystem::path& filename);

}  // namespace splat
