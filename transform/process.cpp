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

#include "process.h"

#include <algorithm>
#include <cmath>
#include <set>


namespace splat {

static std::unique_ptr<DataTable> filter(const DataTable* dataTable,
                                         std::function<bool(const Row&, size_t)> predicate) {
  std::vector<uint32_t> indices;
  const size_t numRows = dataTable->getNumRows();
  indices.reserve(numRows);

  size_t index = 0;
  Row row;
  for (size_t i = 0; i < dataTable->getNumRows(); i++) {
    dataTable->getRow(i, row);
    if (predicate && predicate(row, i)) {
      indices.push_back(static_cast<uint32_t>(i));
    }
  }

  return dataTable->permuteRows(indices);
}

std::unique_ptr<DataTable> processDataTable(DataTable* dataTable, const std::vector<ProcessAction>& processActions) {
  assert(dataTable);
  std::unique_ptr<DataTable> result;
  result.reset(dataTable);


  for (size_t i = 0; i < processActions.size(); i++) {
  }

  return result;
}

}  // namespace splat
