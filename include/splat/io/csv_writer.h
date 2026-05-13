/**
 * @file csv_writer.h
 * @brief CSV file export functionality for Gaussian datasets.
 *
 * This file provides utilities for writing DataTable contents to
 * comma-separated value (CSV) formatted files.
 */

#pragma once

#include <filesystem>

namespace splat {

class DataTable;

void writeCSV(const std::filesystem::path& path, DataTable* dataTable);

}  // namespace splat
