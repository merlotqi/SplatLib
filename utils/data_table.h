#pragma once

#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

using TypedArray = std::variant<std::vector<int8_t>, std::vector<uint8_t>, std::vector<int16_t>, std::vector<uint16_t>,
                                std::vector<int32_t>, std::vector<uint32_t>, std::vector<float>, std::vector<double>>;

enum class ColumnType {
  INT8,
  UINT8,
  INT16,
  UINT16,
  INT32,
  UINT32,
  FLOAT32,
  FLOAT64,
};

inline ColumnType get_column_type(const TypedArray& data) {
  return std::visit(
      [](auto&& arg) -> ColumnType {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::vector<int8_t>>) {
          return ColumnType::INT8;
        } else if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
          return ColumnType::UINT8;
        } else if constexpr (std::is_same_v<T, std::vector<int16_t>>) {
          return ColumnType::INT16;
        } else if constexpr (std::is_same_v<T, std::vector<uint16_t>>) {
          return ColumnType::UINT16;
        } else if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
          return ColumnType::INT32;
        } else if constexpr (std::is_same_v<T, std::vector<uint32_t>>) {
          return ColumnType::UINT32;
        } else if constexpr (std::is_same_v<T, std::vector<float>>) {
          return ColumnType::FLOAT32;
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
          return ColumnType::FLOAT64;
        }
      },
      data);
}

struct Column {
  std::string name;
  TypedArray data;

  Column(const std::string_view& name, const TypedArray& data) : name(name), data(data) {}

  ColumnType data_type() const { return get_column_type(data); }

  size_t size() const {
    return std::visit([](auto&& arg) -> size_t { return arg.size(); }, data);
  }

  template <typename T>
  T value_as(size_t index) const {
    return std::visit(
        [index](auto&& arg) -> T {
          if (index >= arg.size()) throw std::out_of_range("Index out of range");

          using SourceType = typename std::decay_t<decltype(arg)>::value_type;

          if constexpr (std::is_same_v<T, SourceType> || std::is_convertible_v<SourceType, T>) {
            return static_cast<T>(arg[index]);
          } else if constexpr (std::is_same_v<T, std::string> && std::is_arithmetic_v<SourceType>) {
            if constexpr (std::is_integral_v<SourceType>) {
              return std::to_string(static_cast<long long>(arg[index]));
            } else if constexpr (std::is_floating_point_v<SourceType>) {
              return std::to_string(static_cast<long double>(arg[index]));
            }
          } else if constexpr (std::is_same_v<SourceType, std::string> && std::is_arithmetic_v<T>) {
            try {
              if constexpr (std::is_integral_v<T>) {
                if constexpr (std::is_signed_v<T>) {
                  return static_cast<T>(std::stoll(arg[index]));
                } else {
                  return static_cast<T>(std::stoull(arg[index]));
                }
              } else if constexpr (std::is_floating_point_v<T>) {
                return static_cast<T>(std::stold(arg[index]));
              }
            } catch (const std::exception&) {
              throw std::runtime_error("Cannot convert string to numeric type");
            }
          } else if constexpr (std::is_same_v<T, std::string> &&
                               (std::is_same_v<SourceType, char> || std::is_same_v<SourceType, const char*>)) {
            return std::string(1, arg[index]);
          } else {
            static_assert(!sizeof(T*), "Unsupported type conversion from column type to requested type");
            return T{};
          }
        },
        data);
  }

  template <typename T>
  void set_value(size_t index, T value) {
    std::visit(
        [index, value](auto&& arg) {
          if (index >= arg.size()) throw std::out_of_range("index out of range");

          using Q = std::decay_t<decltype(arg)>::value_type;
          static_assert(std::is_convertible_v<T, Q>, "Cannot set value: type mismatch between input and column type");
          arg[index] = static_cast<Q>(value);
        },
        data);
  }
};

using Row = std::map<std::string, double>;

struct DataTable {
  std::vector<Column> columns;

  DataTable(const std::vector<Column>& columns) : columns(std::move(columns)) {
    if (this->columns.empty()) {
      throw std::runtime_error("DataTable must have at least one column");
    }

    const size_t first_size = this->columns[0].size();
    for (size_t i = 1; i < this->columns.size(); i++) {
      if (this->columns[i].size() != first_size) {
        throw std::runtime_error("Column '" + this->columns[i].name + "' has inconsistent number of rows: expected " +
                                 std::to_string(first_size) + ", got " + std::to_string(this->columns[i].size()));
      }
    }
  }

  size_t row_size() const {
    if (this->columns.empty()) return 0;
    return this->columns[0].size();
  }

  size_t column_size() const { return this->columns.size(); }

  Row row(size_t index) const {
    if (index >= row_size()) {
      throw std::out_of_range("index out of range");
    }
    Row row;
    for (const auto& column : this->columns) {
      row[column.name] = column.value_as<double>(index);
    }
    return row;
  }

  void row(size_t index, Row& row) {
    if (index >= row_size()) {
      throw std::out_of_range("index out of range");
    }
    row.clear();
    for (const auto& column : this->columns) {
      row[column.name] = column.value_as<double>(index);
    }
  }

  void set_row(size_t index, const Row& row) {
    if (index >= row_size()) {
      throw std::out_of_range("index out of range");
    }

    for (auto& column : columns) {
      auto it = row.find(column.name);
      if (it != row.end()) {
        column.set_value(index, it->second);
      }
    }
  }

  std::vector<std::string> get_column_names() const {
    std::vector<std::string> names;
    for (const auto& column : columns) {
      names.push_back(column.name);
    }
    return names;
  }

  bool remove_column(const std::string& name) {
    auto it =
        std::find_if(columns.begin(), columns.end(), [&name](const Column& column) { return column.name == name; });
    if (it != columns.end()) {
      columns.erase(it);
      return true;
    }
    return false;
  }

  template <typename T>
  T value_at(size_t row, size_t col) const {
    return columns[col].value_as<T>(row);
  }
};