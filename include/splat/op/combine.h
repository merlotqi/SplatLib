/**
 * @file splat/op/combine.h
 * @brief Merge multiple `SplatCloud` instances into one.
 *
 */

#pragma once

#include <memory>
#include <vector>

namespace splat {

class SplatCloud;

/**
 * @brief Merges multiple DataTables into a single combined table.
 *
 * Creates a union of columns (by name and type) and concatenates rows.
 * Input tables are consumed (moved from) during the operation.
 *
 * @param dataTables Tables to combine (will be emptied)
 * @return Merged table, or nullptr if input is empty
 */
std::unique_ptr<SplatCloud> combine(std::vector<std::unique_ptr<SplatCloud>>& dataTables);

}  // namespace splat
