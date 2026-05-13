#include <splat/models/data-table.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <variant>
#include <vector>

namespace splat {

DataTable::DataTable(const std::vector<Column>& columns) {
  if (columns.empty()) {
    throw std::runtime_error("DataTable must have at least one column");
  }

  const size_t expected_length = columns[0].length();
  for (size_t i = 1; i < columns.size(); ++i) {
    if (columns[i].length() != expected_length) {
      throw std::runtime_error("Column '" + columns[i].name + "' has inconsistent number of rows: expected " +
                               std::to_string(expected_length) + ", got " + std::to_string(columns[i].length()));
    }
  }
  this->columns = std::move(columns);
}

size_t DataTable::getNumRows() const {
  if (columns.empty()) {
    return 0;
  }
  return columns[0].length();
}

Row DataTable::getRow(size_t index, const std::vector<int>& columnIdx) const {
  if (index >= getNumRows()) {
    throw std::out_of_range("index out of range");
  }
  Row row;

  if (columnIdx.empty()) {
    for (const auto& column : columns) {
      row[column.name] = column.getValue(index);
    }
  } else {
    for (const auto& idx : columnIdx) {
      if (idx < 0 || static_cast<size_t>(idx) >= columns.size()) {
        throw std::out_of_range("column index out of range");
      }
      const auto& column = columns[idx];
      row[column.name] = column.getValue(index);
    }
  }
  return row;
}

void DataTable::getRow(size_t index, Row& row, const std::vector<int>& columnIdx) const {
  if (index >= getNumRows()) {
    throw std::out_of_range("index out of range");
  }
  if (columnIdx.empty()) {
    for (const auto& column : columns) {
      row[column.name] = column.getValue(index);
    }
  } else {
    for (const auto& idx : columnIdx) {
      if (idx < 0 || static_cast<size_t>(idx) >= columns.size()) {
        throw std::out_of_range("column index out of range");
      }
      const auto& column = columns[idx];
      row[column.name] = column.getValue(index);
    }
  }
}

void DataTable::setRow(size_t index, const Row& row) {
  if (index >= getNumRows()) {
    throw std::out_of_range("Row index out of bounds in setRow");
  }
  for (auto&& column : columns) {
    auto it = row.find(column.name);
    if (it != row.end()) {
      column.setValue(index, it->second);
    }
  }
}

size_t DataTable::getNumColumns() const { return columns.size(); }

std::vector<std::string> DataTable::getColumnNames() const {
  std::vector<std::string> names;
  for (const auto& column : columns) {
    names.push_back(column.name);
  }
  return names;
}

std::vector<ColumnType> DataTable::getColumnTypes() const {
  std::vector<ColumnType> types;
  for (const auto& column : columns) {
    types.push_back(column.getType());
  }
  return types;
}

const Column& DataTable::getColumn(size_t index) const {
  if (index >= columns.size()) {
    throw std::out_of_range("Column index out of bounds in getColumn");
  }
  return columns[index];
}

Column& DataTable::getColumn(size_t index) {
  if (index >= columns.size()) {
    throw std::out_of_range("Column index out of bounds in getColumn");
  }
  return columns[index];
}

int DataTable::getColumnIndex(const std::string& name) const {
  for (size_t i = 0; i < columns.size(); ++i) {
    if (columns[i].name == name) {
      return (int)i;
    }
  }
  return -1;
}

const Column& DataTable::getColumnByName(const std::string& name) const {
  int index = getColumnIndex(name);
  if (index == -1) {
    throw std::out_of_range("Column not found: " + name);
  }
  return columns[index];
}

Column& DataTable::getColumnByName(const std::string& name) {
  int index = getColumnIndex(name);
  if (index == -1) {
    throw std::out_of_range("Column not found: " + name);
  }
  return columns[index];
}

bool DataTable::hasColumn(const std::string& name) const { return getColumnIndex(name) != -1; }

void DataTable::addColumn(const Column& column) {
  if (columns.size() > 0 && column.length() != getNumRows()) {
    throw std::runtime_error("Column '" + column.name + "' has inconsistent number of rows: expected " +
                             std::to_string(getNumRows()) + ", got " + std::to_string(column.length()));
  }
  columns.push_back(std::move(column));
}

bool DataTable::removeColumn(const std::string& name) {
  auto it = std::remove_if(columns.begin(), columns.end(), [&name](const auto& col) { return col.name == name; });
  if (it == columns.end()) {
    return false;
  }
  columns.erase(it, columns.end());
  return true;
}

std::unique_ptr<DataTable> DataTable::clone(const std::vector<std::string>& columnNames) const {
  std::vector<Column> cloned_cols;
  if (columnNames.empty()) {
    cloned_cols.reserve(columns.size());
    for (const auto& col : columns) {
      TypedArray cloned_data = std::visit([](const auto& vec) -> TypedArray { return vec; }, col.data);
      cloned_cols.emplace_back(Column{col.name, std::move(cloned_data)});
    }
  } else {
    cloned_cols.reserve(columnNames.size());
    for (const auto& name : columnNames) {
      if (this->hasColumn(name)) {
        const auto& col = this->getColumnByName(name);
        TypedArray cloned_data = std::visit([](const auto& vec) -> TypedArray { return vec; }, col.data);
        cloned_cols.emplace_back(Column{col.name, std::move(cloned_data)});
      } else {
        throw std::runtime_error("Column not found: " + name);
      }
    }
  }
  return std::make_unique<DataTable>(cloned_cols);
}

std::unique_ptr<DataTable> DataTable::permuteRows(const std::vector<uint32_t>& indices) const {
  std::vector<Column> new_columns;
  new_columns.reserve(columns.size());
  size_t new_length = indices.size();
  size_t old_len = getNumRows();

  for (const auto& old_col : columns) {
    TypedArray new_data = std::visit(
        [&indices, new_length, old_len](const auto& old_vec) -> TypedArray {
          using T = typename std::decay_t<decltype(old_vec)>::value_type;
          std::vector<T> new_vec(new_length);

          for (size_t j = 0; j < new_length; j++) {
            size_t src_index = indices[j];
            if (src_index >= old_len) {
              throw std::out_of_range("Permutation index out of bounds.");
            }
            new_vec[j] = old_vec[src_index];
          }
          return new_vec;
        },
        old_col.data);

    new_columns.emplace_back(Column{old_col.name, std::move(new_data)});
  }

  return std::make_unique<DataTable>(new_columns);
}

}  // namespace splat
