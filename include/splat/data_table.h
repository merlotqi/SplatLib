/**
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
 */

#pragma once

#include <absl/types/span.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>


namespace splat {

using Row = std::map<std::string, float>;

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

using TypedArray = std::variant<std::vector<int8_t>,    // Int8Array
                                std::vector<uint8_t>,   // Uint8Array
                                std::vector<int16_t>,   // Int16Array
                                std::vector<uint16_t>,  // Uint16Array
                                std::vector<int32_t>,   // Int32Array
                                std::vector<uint32_t>,  // Uint32Array
                                std::vector<float>,     // Float32Array
                                std::vector<double>     // Float64Array
                                >;
struct Column {
  std::string name;
  TypedArray data;

  ColumnType getType() const { return static_cast<ColumnType>(data.index()); }

  size_t length() const {
    return std::visit([](const auto& vec) -> size_t { return vec.size(); }, data);
  }

  template <typename T>
  const std::vector<T>& asVector() const {
    return std::get<std::vector<T>>(data);
  }

  template <typename T>
  std::vector<T>& asVector() {
    return std::get<std::vector<T>>(data);
  }

  template <typename T>
  const absl::Span<const T> asSpan() const {
    return absl::MakeConstSpan(asVector<T>());
  }

  template <typename T>
  absl::Span<T> asSpan() {
    return absl::MakeSpan(asVector<T>());
  }

  template <typename T>
  T getValue(size_t index) const {
    if (index >= length()) throw std::out_of_range("Index out of range");

    return std::visit(
        [index](const auto& vec) -> T {
          if constexpr (std::is_same_v<T, std::string>) {
            return std::to_string(vec[index]);
          } else {
            return static_cast<T>(vec[index]);
          }
        },
        data);
  }

  template <typename T>
  void setValue(size_t index, T value) {
    if (index >= length()) {
      if (index >= length()) throw std::out_of_range("Index out of range");
    }

    auto visitor = [index, value](auto& vec) {
      using VectorType = std::decay_t<decltype(vec)>;
      using Q = typename VectorType::value_type;

      if constexpr (std::is_same_v<T, std::string>) {
        // Handle string input
        const std::string& str_value = value;
        try {
          if constexpr (std::is_same_v<Q, int8_t> || std::is_same_v<Q, int16_t> || std::is_same_v<Q, int32_t>) {
            // For signed integers
            long long temp = std::stoll(str_value);
            if (temp > static_cast<long long>(std::numeric_limits<Q>::max()) ||
                temp < static_cast<long long>(std::numeric_limits<Q>::min())) {
              throw std::range_error("String value out of range for signed integer type.");
            }
            vec[index] = static_cast<Q>(temp);
          } else if constexpr (std::is_same_v<Q, uint8_t> || std::is_same_v<Q, uint16_t> ||
                               std::is_same_v<Q, uint32_t>) {
            // For unsigned integers
            unsigned long long temp = std::stoull(str_value);
            if (temp > static_cast<unsigned long long>(std::numeric_limits<Q>::max())) {
              throw std::range_error("String value out of range for unsigned integer type.");
            }
            vec[index] = static_cast<Q>(temp);
          } else if constexpr (std::is_same_v<Q, float>) {
            vec[index] = std::stof(str_value);
          } else if constexpr (std::is_same_v<Q, double>) {
            vec[index] = std::stod(str_value);
          } else {
            throw std::runtime_error("Unsupported type for string conversion.");
          }
        } catch (const std::invalid_argument&) {
          throw std::runtime_error("Invalid argument for string to number conversion.");
        } catch (const std::out_of_range&) {
          throw std::range_error("String value out of range.");
        }
      } else {
        // Handle non-string input
        using ValueType = std::decay_t<T>;

        if constexpr (!std::is_convertible_v<ValueType, Q>) {
          throw std::runtime_error("Input type cannot be converted to internal column type.");
        }

        // Check for potential overflow/truncation
        if constexpr (std::is_integral_v<Q>) {
          // Check if value fits in internal type
          if constexpr (std::is_integral_v<ValueType>) {
            // Both are integral types
            if (value > static_cast<ValueType>(std::numeric_limits<Q>::max()) ||
                value < static_cast<ValueType>(std::numeric_limits<Q>::min())) {
              throw std::range_error("Value exceeds range of internal integer type.");
            }
          } else if constexpr (std::is_floating_point_v<ValueType>) {
            // Floating point to integer conversion
            // Check for overflow
            if (value > static_cast<ValueType>(std::numeric_limits<Q>::max()) ||
                value < static_cast<ValueType>(std::numeric_limits<Q>::min())) {
              throw std::range_error("Value exceeds range of internal integer type.");
            }
            // Check for truncation
            if (std::abs(value - std::round(value)) > std::numeric_limits<ValueType>::epsilon() * 10) {
              throw std::range_error("Floating-point value cannot be exactly represented in internal integer type.");
            }
          }
        } else if constexpr (std::is_floating_point_v<Q>) {
          // Floating point to floating point conversion
          if constexpr (std::is_same_v<Q, float> && std::is_same_v<ValueType, double>) {
            // double to float conversion - check for overflow
            if (std::abs(value) > std::numeric_limits<float>::max()) {
              throw std::range_error("Double value exceeds float range.");
            }
          }
        }

        // Safe to assign
        vec[index] = static_cast<Q>(value);
      }
    };

    std::visit(visitor, data);
  }

  float getValue(size_t index) const { return getValue<float>(index); }

  void setValue(size_t index, float value) { setValue<float>(index, value); }

  size_t bytePreElement() const {
    return std::visit(
        [](const auto& vec) -> size_t {
          using T = typename std::decay_t<decltype(vec)>::value_type;
          return sizeof(T);
        },
        data);
  }

  size_t totalByteSize() const { return length() * bytePreElement(); }

  const uint8_t* rawPointer() const {
    return std::visit([](const auto& vec) -> const uint8_t* { return reinterpret_cast<const uint8_t*>(vec.data()); },
                      data);
  }

  uint8_t* rawPointer() {
    return std::visit([](auto& vec) -> uint8_t* { return reinterpret_cast<uint8_t*>(vec.data()); }, data);
  }

  template <typename T>
  bool every(T value) const {
    return std::visit(
        [&](const auto& vec) -> bool {
          using Q = typename std::decay_t<decltype(vec)>::value_type;

          Q target;
          if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, const char*>) {
            target = parseString<Q>(value);
          } else {
            target = static_cast<Q>(value);
          }

          if constexpr (std::is_floating_point_v<Q>) {
            const Q eps = static_cast<Q>(1e-10);
            return std::all_of(vec.begin(), vec.end(), [target, eps](Q x) { return std::abs(x - target) < eps; });
          } else {
            return std::all_of(vec.begin(), vec.end(), [target](Q x) { return x == target; });
          }
        },
        data);
  }

  template <typename T>
  bool some(T value) const {
    return std::visit(
        [&](const auto& vec) -> bool {
          using Q = typename std::decay_t<decltype(vec)>::value_type;

          Q target;
          if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, const char*>) {
            target = parseString<Q>(value);
          } else {
            target = static_cast<Q>(value);
          }
          if constexpr (std::is_floating_point_v<Q>) {
            const Q eps = static_cast<Q>(1e-10);
            return std::any_of(vec.begin(), vec.end(), [target, eps](Q x) { return std::abs(x - target) < eps; });
          } else {
            return std::any_of(vec.begin(), vec.end(), [target](Q x) { return x == target; });
          }
        },
        data);
  }

  template <typename Q, typename T>
  Q parseString(T value) const {
    std::string s;
    if constexpr (std::is_same_v<T, const char*>)
      s = value;
    else
      s = value;

    try {
      if constexpr (std::is_floating_point_v<Q>) return static_cast<Q>(std::stod(s));
      if constexpr (std::is_signed_v<Q>) return static_cast<Q>(std::stoll(s));
      if constexpr (std::is_unsigned_v<Q>) return static_cast<Q>(std::stoull(s));
    } catch (...) {
      throw std::runtime_error("Column comparison: String conversion failed");
    }
    return Q{};
  }
};

class DataTable {
 public:
  std::vector<Column> columns;

  DataTable() = default;
  DataTable(const std::vector<Column>& columns);

  DataTable(const DataTable& other) = delete;
  DataTable& operator=(const DataTable& other) = delete;
  DataTable(DataTable&& other) noexcept = default;
  DataTable& operator=(DataTable&& other) noexcept = default;

  size_t getNumRows() const;
  Row getRow(size_t index, const std::vector<int>& columnIdx = {}) const;
  void getRow(size_t index, Row& row, const std::vector<int>& columnIdx = {}) const;
  void setRow(size_t index, const Row& row);
  size_t getNumColumns() const;

  std::vector<std::string> getColumnNames() const;
  std::vector<ColumnType> getColumnTypes() const;
  const Column& getColumn(size_t index) const;
  Column& getColumn(size_t index);
  int getColumnIndex(const std::string& name) const;
  const Column& getColumnByName(const std::string& name) const;
  Column& getColumnByName(const std::string& name);

  bool hasColumn(const std::string& name) const;
  void addColumn(const Column& column);
  bool removeColumn(const std::string& name);

  std::unique_ptr<DataTable> clone(const std::vector<std::string>& columnNames = {}) const;
  std::unique_ptr<DataTable> permuteRows(const std::vector<uint32_t>& indices) const;
};

void generateOrdering(const DataTable* dataTable, absl::Span<uint32_t> indices);

void transform(DataTable* dataTable, const Eigen::Vector3f& t, const Eigen::Quaternionf& r, float s);

}  // namespace splat
