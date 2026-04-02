/***********************************************************************************
 *
 * splat - Gaussian splat decimation (NanoGS-style progressive merge).
 * Ported from splat-transform lib/data-table/decimate.ts.
 *
 ***********************************************************************************/

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace splat {

class DataTable;

/**
 * @brief Sort row indices by visibility score (sigmoid(opacity) * exp(scale sum)), descending.
 * @param data_table Must have opacity, scale_0, scale_1, scale_2 columns.
 * @param indices In/out: length = numRows, typically 0..n-1; reordered on success.
 */
void sortByVisibility(const DataTable& data_table, std::vector<uint32_t>& indices);

/**
 * @brief Simplify splat count toward target using pairwise merging (KHR / NanoGS-style).
 * @param data_table Input splats
 * @param targetCount Desired output row count (<=0 yields empty table with same columns)
 * @return New table; unchanged clone if already <= targetCount
 */
std::unique_ptr<DataTable> simplifyGaussians(const DataTable& data_table, int targetCount);

}  // namespace splat
