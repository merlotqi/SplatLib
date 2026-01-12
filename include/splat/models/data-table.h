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

#pragma once

#include <absl/types/span.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

/**
 * @file datatable.h
 * @brief Data table implementation for typed columnar storage and manipulation
 */

namespace splat {

/**
 * @brief Represents a single row of data as a map of column names to float values
 */
using Row = std::map<std::string, float>;

/**
 * @brief Enumeration of supported column data types
 */
enum class ColumnType : std::uint8_t {
  INT8,     ///< Signed 8-bit integer
  UINT8,    ///< Unsigned 8-bit integer
  INT16,    ///< Signed 16-bit integer
  UINT16,   ///< Unsigned 16-bit integer
  INT32,    ///< Signed 32-bit integer
  UINT32,   ///< Unsigned 32-bit integer
  FLOAT32,  ///< 32-bit floating point
  FLOAT64,  ///< 64-bit floating point
};

/**
 * @brief Variant type representing different typed array storage options
 */
using TypedArray = std::variant<std::vector<int8_t>,    // Int8Array
                                std::vector<uint8_t>,   // Uint8Array
                                std::vector<int16_t>,   // Int16Array
                                std::vector<uint16_t>,  // Uint16Array
                                std::vector<int32_t>,   // Int32Array
                                std::vector<uint32_t>,  // Uint32Array
                                std::vector<float>,     // Float32Array
                                std::vector<double>     // Float64Array
                                >;

/**
 * @brief Represents a single column of typed data with metadata
 *
 * A Column contains a name and typed data storage using std::variant.
 * Provides type-safe access, value retrieval, and modification operations.
 */
struct Column {
  std::string name;  ///< Column name identifier
  TypedArray data;   ///< Typed data storage using variant

  /**
   * @brief Get the column's data type
   * @return ColumnType corresponding to the current variant index
   */
  ColumnType getType() const { return static_cast<ColumnType>(data.index()); }

  /**
   * @brief Get the number of elements in the column
   * @return Number of elements in the underlying vector
   */
  size_t length() const {
    return std::visit([](const auto& vec) -> size_t { return vec.size(); }, data);
  }

  /**
   * @brief Get const reference to underlying vector of specified type
   * @tparam T Type of vector to retrieve
   * @return Const reference to std::vector<T>
   * @throws std::bad_variant_access if variant doesn't hold requested type
   */
  template <typename T>
  const std::vector<T>& asVector() const {
    return std::get<std::vector<T>>(data);
  }

  /**
   * @brief Get reference to underlying vector of specified type
   * @tparam T Type of vector to retrieve
   * @return Reference to std::vector<T>
   * @throws std::bad_variant_access if variant doesn't hold requested type
   */
  template <typename T>
  std::vector<T>& asVector() {
    return std::get<std::vector<T>>(data);
  }

  /**
   * @brief Get const span view of column data
   * @tparam T Element type of the span
   * @return Const span view of the data
   */
  template <typename T>
  const absl::Span<const T> asSpan() const {
    return absl::MakeConstSpan(asVector<T>());
  }

  /**
   * @brief Get mutable span view of column data
   * @tparam T Element type of the span
   * @return Mutable span view of the data
   */
  template <typename T>
  absl::Span<T> asSpan() {
    return absl::MakeSpan(asVector<T>());
  }

  /**
   * @brief Get value at specified index with type conversion
   * @tparam T Desired return type
   * @param index Element index
   * @return Value converted to type T
   * @throws std::out_of_range if index is out of bounds
   * @note For string conversion, returns string representation via std::to_string
   */
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

  /**
   * @brief Set value at specified index with type conversion and range checking
   * @tparam T Input value type
   * @param index Element index
   * @param value Value to set
   * @throws std::out_of_range if index is out of bounds
   * @throws std::runtime_error if input type cannot be converted
   * @throws std::range_error if value exceeds internal type range or causes truncation
   *
   * Performs comprehensive type conversion checking including:
   * - Integer overflow detection
   * - Floating point to integer truncation detection
   * - Double to float overflow detection
   */
  template <typename T>
  void setValue(size_t index, T value) {
    if (index >= length()) {
      if (index >= length()) throw std::out_of_range("Index out of range");
    }

    auto visitor = [index, value](auto& vec) {
      using VectorType = std::decay_t<decltype(vec)>;
      using Q = typename VectorType::value_type;

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
    };

    std::visit(visitor, data);
  }

  /**
   * @brief Convenience method to get float value
   * @param index Element index
   * @return Value as float
   */
  float getValue(size_t index) const { return getValue<float>(index); }

  /**
   * @brief Convenience method to set float value
   * @param index Element index
   * @param value Float value to set
   */
  void setValue(size_t index, float value) { setValue<float>(index, value); }

  /**
   * @brief Get bytes per element based on column type
   * @return Size of each element in bytes
   */
  size_t bytePreElement() const {
    return std::visit(
        [](const auto& vec) -> size_t {
          using T = typename std::decay_t<decltype(vec)>::value_type;
          return sizeof(T);
        },
        data);
  }

  /**
   * @brief Calculate total memory usage of column data
   * @return Total bytes used by column data
   */
  size_t totalByteSize() const { return length() * bytePreElement(); }

  /**
   * @brief Get const raw pointer to column data
   * @return Const pointer to raw byte data
   */
  const uint8_t* rawPointer() const {
    return std::visit([](const auto& vec) -> const uint8_t* { return reinterpret_cast<const uint8_t*>(vec.data()); },
                      data);
  }

  /**
   * @brief Get mutable raw pointer to column data
   * @return Pointer to raw byte data
   */
  uint8_t* rawPointer() {
    return std::visit([](auto& vec) -> uint8_t* { return reinterpret_cast<uint8_t*>(vec.data()); }, data);
  }

  /**
   * @brief Check if all elements equal a given value
   * @tparam T Type of comparison value
   * @param value Value to compare against
   * @return true if all elements equal the value, false otherwise
   *
   * For floating point comparisons, uses epsilon-based equality (1e-10).
   * String values are parsed to the column's native type.
   */
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

  /**
   * @brief Check if any element equals a given value
   * @tparam T Type of comparison value
   * @param value Value to compare against
   * @return true if at least one element equals the value, false otherwise
   *
   * For floating point comparisons, uses epsilon-based equality (1e-10).
   * String values are parsed to the column's native type.
   */
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

  /**
   * @brief Parse string to column's native type
   * @tparam Q Column's native type
   * @tparam T Input type (string or const char*)
   * @param value String value to parse
   * @return Parsed value of type Q
   * @throws std::runtime_error if string conversion fails
   */
  template <typename Q, typename T>
  Q parseString(T value) const {
    std::string s = value;

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

/**
 * @brief Tabular data structure with typed columns
 *
 * DataTable provides a collection of typed Column objects with operations
 * for data manipulation, querying, and transformation. Supports move semantics
 * but disables copying to prevent accidental deep copies.
 */
class DataTable {
 public:
  std::vector<Column> columns;  ///< Collection of column data

  /**
   * @brief Default constructor
   */
  DataTable() = default;

  /**
   * @brief Construct from existing columns
   * @param columns Vector of columns to initialize with
   */
  DataTable(const std::vector<Column>& columns);

  // Disable copy semantics to prevent accidental deep copies
  DataTable(const DataTable& other) = delete;
  DataTable& operator=(const DataTable& other) = delete;

  // Enable move semantics
  DataTable(DataTable&& other) noexcept = default;
  DataTable& operator=(DataTable&& other) noexcept = default;

  /**
   * @brief Get number of rows in the table
   * @return Number of rows (0 if no columns)
   * @note Assumes all columns have equal length
   */
  size_t getNumRows() const;

  /**
   * @brief Get a row as a map of column names to values
   * @param index Row index
   * @param columnIdx Optional indices of specific columns to include
   * @return Row as map<string, float>
   */
  Row getRow(size_t index, const std::vector<int>& columnIdx = {}) const;

  /**
   * @brief Get a row into existing map
   * @param index Row index
   * @param row Output map to populate
   * @param columnIdx Optional indices of specific columns to include
   */
  void getRow(size_t index, Row& row, const std::vector<int>& columnIdx = {}) const;

  /**
   * @brief Set values for a specific row
   * @param index Row index
   * @param row Map of column names to values
   */
  void setRow(size_t index, const Row& row);

  /**
   * @brief Get number of columns in the table
   * @return Number of columns
   */
  size_t getNumColumns() const;

  /**
   * @brief Get all column names
   * @return Vector of column names
   */
  std::vector<std::string> getColumnNames() const;

  /**
   * @brief Get data types of all columns
   * @return Vector of ColumnType values
   */
  std::vector<ColumnType> getColumnTypes() const;

  /**
   * @brief Get const reference to column by index
   * @param index Column index
   * @return Const reference to Column
   */
  const Column& getColumn(size_t index) const;

  /**
   * @brief Get mutable reference to column by index
   * @param index Column index
   * @return Reference to Column
   */
  Column& getColumn(size_t index);

  /**
   * @brief Get column index by name
   * @param name Column name to find
   * @return Column index or -1 if not found
   */
  int getColumnIndex(const std::string& name) const;

  /**
   * @brief Get const column by name
   * @param name Column name
   * @return Const reference to Column
   * @throws std::runtime_error if column not found
   */
  const Column& getColumnByName(const std::string& name) const;

  /**
   * @brief Get mutable column by name
   * @param name Column name
   * @return Reference to Column
   * @throws std::runtime_error if column not found
   */
  Column& getColumnByName(const std::string& name);

  /**
   * @brief Check if column exists by name
   * @param name Column name to check
   * @return true if column exists, false otherwise
   */
  bool hasColumn(const std::string& name) const;

  /**
   * @brief Add a new column to the table
   * @param column Column to add
   * @throws std::runtime_error if column length doesn't match existing columns
   */
  void addColumn(const Column& column);

  /**
   * @brief Remove column by name
   * @param name Column name to remove
   * @return true if column was removed, false if not found
   */
  bool removeColumn(const std::string& name);

  /**
   * @brief Create a deep copy of the table
   * @param columnNames Optional subset of columns to clone
   * @return Unique pointer to cloned DataTable
   */
  std::unique_ptr<DataTable> clone(const std::vector<std::string>& columnNames = {}) const;

  /**
   * @brief Create new table with rows permuted according to indices
   * @param indices Permutation indices
   * @return Unique pointer to permuted DataTable
   */
  std::unique_ptr<DataTable> permuteRows(const std::vector<uint32_t>& indices) const;
};

/**
 * @brief Generate row ordering based on data table contents
 * @param dataTable Source data table
 * @param indices Output span to fill with ordering indices
 */
void generateOrdering(const DataTable* dataTable, absl::Span<uint32_t> indices);

/**
 * @brief Apply transformation to data table
 * @param dataTable Table to transform
 * @param t Translation vector
 * @param r Rotation quaternion
 * @param s Scale factor
 *
 * Applies translation, rotation, and scaling transformation to relevant
 * columns in the data table.
 */
void transform(DataTable* dataTable, const Eigen::Vector3f& t, const Eigen::Quaternionf& r, float s);

/**
 * @brief Merges multiple DataTables into a single combined table.
 * 
 * Creates a union of columns (by name and type) and concatenates rows.
 * Input tables are consumed (moved from) during the operation.
 * 
 * @param dataTables Tables to combine (will be emptied)
 * @return Merged table, or nullptr if input is empty
 */
std::unique_ptr<DataTable> combine(std::vector<std::unique_ptr<DataTable>>& dataTables);

}  // namespace splat
