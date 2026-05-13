#include "process.h"

#include <cmath>


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
