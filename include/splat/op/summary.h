/***********************************************************************************
 *
 * splat - A C++ library for reading and writing 3D Gaussian Splatting (splat) files.
 *
 * This library provides functionality to convert, manipulate, and process
 * 3D Gaussian splatting data formats used in real-time neural rendering.
 *
 * This file is part of splat.
 *
 * splat is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * splat is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * For more information, visit the project's homepage or contact the author.
 *
 ***********************************************************************************/

#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace splat {

/**
 * @struct ColumnStats
 * @brief Statistical summary for a single data column.
 * * This structure stores descriptive statistics, error counts, and
 * visual distribution data for a specific column in a dataset.
 */
struct ColumnStats {
  /** @brief Minimum value (excluding NaN and Infinity). */
  float min{0.0f};

  /** @brief Maximum value (excluding NaN and Infinity). */
  float max{0.0f};

  /** @brief Median value (the middle value of the sorted data). */
  float median{0.0f};

  /** @brief Arithmetic mean (average) of the values. */
  float mean{0.0f};

  /** @brief Standard deviation, representing the amount of variation or dispersion. */
  float stdDev{0.0f};

  /** @brief Total count of Not-a-Number (NaN) occurrences. */
  std::size_t nanCount{0};

  /** @brief Total count of positive or negative Infinity occurrences. */
  std::size_t infCount{0};

  /** @brief ASCII-based histogram representing the value distribution. */
  std::string histogram;
};

/**
 * @struct SummaryData
 * @brief Statistical summary for an entire DataTable.
 * * Encapsulates metadata about the table and a collection of per-column
 * statistics identified by their column names.
 */
struct SummaryData {
  /** @brief Summary format version, useful for backward compatibility during serialization. */
  std::uint32_t version{1};

  /** @brief Total number of rows processed in the DataTable. */
  std::size_t rowCount{0};

  /** * @brief Per-column statistics keyed by column name.
   * * Maps the column identifier (string) to its corresponding ColumnStats structure.
   */
  std::map<std::string, ColumnStats> columns;
};

class DataTable;

SummaryData computeSummary(const DataTable *dataTable);

}  // namespace splat
