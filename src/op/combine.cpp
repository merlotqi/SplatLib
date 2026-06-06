#include <splat/models/splatcloud.h>

#include <cstring>

namespace splat {

std::unique_ptr<SplatCloud> combine(std::vector<std::unique_ptr<SplatCloud>>& dataTables) {
  if (dataTables.empty()) {
    return nullptr;
  }

  if (dataTables.size() == 1) {
    return std::move(dataTables[0]);
  }

  auto findMatchingColumn = [](const std::vector<Column>& columns, const Column& column) {
    for (int i = 0; i < columns.size(); ++i) {
      if (columns[i].name == column.name && columns[i].getType() == column.getType()) {
        return i;
      }
    }
    return -1;
  };

  // make unique list of columns where name and type much match
  auto&& columns = dataTables[0]->columns;
  for (int i = 1; i < dataTables.size(); ++i) {
    const auto& dataTable = dataTables[i];
    for (int j = 0; j < dataTable->getNumColumns(); ++j) {
      if (-1 == findMatchingColumn(columns, dataTable->columns[j])) {
        columns.push_back(dataTable->columns[j]);
      }
    }
  }

  // count total number of rows
  size_t totalRows = 0;
  for (auto&& dt : dataTables) {
    totalRows += dt->getNumRows();
  }

  // construct output dataTable
  std::vector<Column> resultColumns;
  for (auto&& col : columns) {
    auto data = std::visit(
        [totalRows](const auto& vec) -> TypedArray {
          using T = typename std::decay_t<decltype(vec)>::value_type;
          return std::vector<T>(totalRows);
        },
        col.data);

    resultColumns.push_back({col.name, data});
  }

  // copy data
  int rowOffset = 0;
  for (int i = 0; i < dataTables.size(); ++i) {
    const auto& dataTable = dataTables[i];

    for (int j = 0; j < dataTable->columns.size(); ++j) {
      const auto column = dataTable->columns[j];
      int idx = findMatchingColumn(resultColumns, column);
      auto& targetColumn = resultColumns[idx];
      std::memcpy(targetColumn.rawPointer() + rowOffset * targetColumn.bytePreElement(), column.rawPointer(),
                  column.length() * column.bytePreElement());
    }
    rowOffset += dataTable->getNumRows();
  }

  return std::make_unique<SplatCloud>(resultColumns);
}

}  // namespace splat
