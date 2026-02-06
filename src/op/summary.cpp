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

#include <splat/models/data-table.h>
#include <splat/op/summary.h>

namespace splat {

static constexpr int NUM_BINS = 16;
static constexpr auto BARS = "▁▂▃▄▅▆▇█";

template <typename T>
T quickSelect(absl::Span<T> arr, size_t k, size_t left, size_t right) {
  while (left < right) {
    // Use median-of-three pivot selection for better performance
    const auto mid = (left + right) / 2;
    if (arr[mid] < arr[left]) {
      const auto t = arr[left];
      arr[left] = arr[mid];
      arr[mid] = t;
    }
    if (arr[right] < arr[left]) {
      const auto t = arr[left];
      arr[left] = arr[right];
      arr[right] = t;
    }
    if (arr[right] < arr[mid]) {
      const auto t = arr[mid];
      arr[mid] = arr[right];
      arr[right] = t;
    }

    const T pivot = arr[mid];
    size_t i = left;
    size_t j = right;

    while (i <= j) {
      while (arr[i] < pivot) i++;
      while (arr[j] > pivot) j--;
      if (i <= j) {
        const T t = arr[i];
        arr[i] = arr[j];
        arr[j] = t;
        i++;
        j--;
      }
    }

    if (k <= j) {
      right = j;
    } else if (k >= i) {
      left = i;
    } else {
      break;
    }
  }
  return arr[k];
}

static ColumnStats computeColumnStats(const Column& column) {
  const auto& data = column.asSpan<float>();
  const size_t len = data.length();

  return ColumnStats();
}

SummaryData computeSummary(const DataTable* dataTable) {
  std::map<std::string, ColumnStats> columns;

  for (auto&& column : dataTable->columns) {
    columns[column.name] = computeColumnStats(column);
  }
  return {1, dataTable->getNumRows(), columns};
}

}  // namespace splat
