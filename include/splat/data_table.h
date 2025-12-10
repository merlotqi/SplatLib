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

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>
#include <Eigen/Dense>

namespace splat {

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

using Row = std::map<std::string, double>;
using TypedArray = std::variant<std::vector<int8_t>,    // Int8Array
                                std::vector<uint8_t>,   // Uint8Array
                                std::vector<int16_t>,   // Int16Array
                                std::vector<uint16_t>,  // Uint16Array
                                std::vector<int32_t>,   // Int32Array
                                std::vector<uint32_t>,  // Uint32Array
                                std::vector<float>,     // Float32Array
                                std::vector<double>     // Float64Array
                                >;

class ColumnBase {
 public:
  std::string name;
  ColumnBase(const std::string& name) : name(name) {}
  virtual ~ColumnBase() = default;

  virtual ColumnType getDataType() const = 0;
  virtual size_t length() const = 0;
  virtual std::unique_ptr<ColumnBase> clone() const = 0;
  virtual double getValue(size_t index) const = 0;
  virtual void setValue(size_t index, double value) = 0;
  virtual void permuteData(const ColumnBase* source_col, const std::vector<uint32_t>& indices) = 0;
};

template <typename T>
class Column : public ColumnBase {
 public:
  std::vector<T> data;
  Column(const std::string& name, const std::vector<T>& data) : ColumnBase(name), data(data) {}
  ColumnType getDataType() const override {
    if constexpr (std::is_same_v<T, int8_t>) {
      return ColumnType::INT8;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
      return ColumnType::UINT8;
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return ColumnType::INT16;
    } else if constexpr (std::is_same_v<T, uint16_t>) {
      return ColumnType::UINT16;
    } else if constexpr (std::is_same_v<T, int32_t>) {
      return ColumnType::INT32;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      return ColumnType::UINT32;
    } else if constexpr (std::is_same_v<T, float>) {
      return ColumnType::FLOAT32;
    } else if constexpr (std::is_same_v<T, double>) {
      return ColumnType::FLOAT64;
    } else {
      static_assert(false, "Unsupported data type");
    }
  }

  std::unique_ptr<ColumnBase> clone() const override { return std::make_unique<Column<T>>(name, data); }

  size_t length() const override { return data.size(); }

  double getValue(size_t index) const override {
    if (index >= data.size()) {
      throw std::out_of_range("index out of range");
    }
    return static_cast<double>(data[index]);
  }

  void setValue(size_t index, double value) override {
    if (index >= data.size()) {
      throw std::out_of_range("index out of range");
    }
    data[index] = static_cast<T>(value);
  }

  std::vector<T>& getData() const { return data; }
  std::vector<T> getData() { return data; }

  void permuteData(const ColumnBase* source_col, const std::vector<uint32_t>& indices) override {
    const Column<T>* old_col = dynamic_cast<const Column<T>*>(source_col);

    if (!old_col) {
      throw std::runtime_error("Permute source column type mismatch.");
    }

    size_t new_length = indices.size();
    this->data.resize(new_length);

    const auto& src_data = old_col->data;
    size_t src_len = src_data.size();

    for (size_t j = 0; j < new_length; j++) {
      size_t src_index = indices[j];
      if (src_index >= src_len) {
        throw std::out_of_range("Permutation index out of bounds.");
      }
      this->data[j] = src_data[src_index];
    }
  }
};

class DataTable {
 public:
  std::vector<std::unique_ptr<ColumnBase>> columns;

  DataTable(std::vector<std::unique_ptr<ColumnBase>> columns);

  DataTable(const DataTable& other) = delete;
  DataTable& operator=(const DataTable& other) = delete;

  DataTable(DataTable&& other) = default;
  DataTable& operator=(DataTable&& other) = default;

  size_t getNumRows() const;
  Row getRow(size_t index) const;
  void setRow(size_t index, const Row& row);
  size_t getNumColumns() const;

  std::vector<std::string> getColumnNames() const;
  std::vector<ColumnType> getColumnTypes() const;
  ColumnBase* getColumn(size_t index) const;
  int getColumnIndex(const std::string& name) const;
  ColumnBase* getColumnByName(const std::string& name) const;

  bool hasColumn(const std::string& name) const;
  void addColumn(std::unique_ptr<ColumnBase> column);
  bool removeColumn(const std::string& name);

  DataTable clone() const;
  DataTable permuteRows(const std::vector<uint32_t>& indices) const;

 private:
  std::unique_ptr<ColumnBase> createEmptyColumn(const std::string& name, ColumnType type, size_t length) const;
};

std::vector<uint32_t>& generateOrdering(DataTable& dataTable, std::vector<uint32_t>& indices);

void transform(DataTable& dataTable, const Eigen::Vector3d& t, const Eigen::Quaterniond& r, float s);

}  // namespace splat
