/**
 * @file csv_writer.h
 * @brief CSV file export functionality for Gaussian datasets.
 *
 * This file provides utilities for writing SplatCloud contents to
 * comma-separated value (CSV) formatted files.
 */

#pragma once

#include <filesystem>

namespace splat {

class SplatCloud;

void writeCSV(const std::filesystem::path& path, SplatCloud* dataTable);

}  // namespace splat
