/**
 * @file splat/io/ksplat_reader.h
 * @brief Read .ksplat container format into SplatCloud
 *
 * Provides functions to load Kiri Engine's compressed Gaussian splat format.
 */

#pragma once

#include <splat/models/splatcloud.h>

#include <filesystem>

namespace splat {

/**
 * @brief Load Gaussian splat data from .ksplat file
 *
 * Reads a .ksplat file (Kiri Engine's compressed format) and decodes
 * it into a SplatCloud with standard Gaussian splat columns.
 *
 * @param filename Path to the .ksplat file
 * @return SplatCloud containing decoded splat data
 */
std::unique_ptr<SplatCloud> readKsplat(const std::filesystem::path& filename);

}  // namespace splat
