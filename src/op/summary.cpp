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
