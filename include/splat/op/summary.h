/**
 * @file summary.h
 * @brief Statistical summary computation for Gaussian data tables.
 *
 * This file provides structures and functions for calculating descriptive
 * statistics, histograms, and data quality metrics for SplatCloud datasets.
 */

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
 * @brief Statistical summary for an entire SplatCloud.
 * * Encapsulates metadata about the table and a collection of per-column
 * statistics identified by their column names.
 */
struct SummaryData {
  /** @brief Summary format version, useful for backward compatibility during serialization. */
  std::uint32_t version{1};

  /** @brief Total number of rows processed in the SplatCloud. */
  std::size_t rowCount{0};

  /** * @brief Per-column statistics keyed by column name.
   * * Maps the column identifier (string) to its corresponding ColumnStats structure.
   */
  std::map<std::string, ColumnStats> columns;
};

class SplatCloud;

SummaryData computeSummary(const SplatCloud* dataTable);

}  // namespace splat
