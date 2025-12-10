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

#include <splat/data_table.h>
#include <splat/maths/rotate-sh.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>

namespace splat {

DataTable::DataTable(std::vector<std::unique_ptr<ColumnBase>> columns) {
  if (columns.empty()) {
    throw std::runtime_error("DataTable must have at least one column");
  }

  const size_t expected_length = columns[0]->length();
  for (size_t i = 1; i < columns.size(); ++i) {
    if (columns[i]->length() != expected_length) {
      throw std::runtime_error("Column '" + columns[i]->name + "' has inconsistent number of rows: expected " +
                               std::to_string(expected_length) + ", got " + std::to_string(columns[i]->length()));
    }
  }
  this->columns = std::move(columns);
}

size_t DataTable::getNumRows() const {
  if (columns.empty()) {
    return 0;
  }
  return columns[0]->length();
}

Row DataTable::getRow(size_t index) const {
  if (index >= getNumRows()) {
    throw std::out_of_range("index out of range");
  }
  Row row;
  for (const auto& column : columns) {
    row[column->name] = column->getValue(index);
  }
  return row;
}

void DataTable::setRow(size_t index, const Row& row) {
  if (index >= getNumRows()) {
    throw std::out_of_range("Row index out of bounds in setRow");
  }
  for (const auto& column : columns) {
    auto it = row.find(column->name);
    if (it != row.end()) {
      column->setValue(index, it->second);
    }
  }
}

size_t DataTable::getNumColumns() const { return columns.size(); }

std::vector<std::string> DataTable::getColumnNames() const {
  std::vector<std::string> names;
  for (const auto& column : columns) {
    names.push_back(column->name);
  }
  return names;
}

std::vector<ColumnType> DataTable::getColumnTypes() const {
  std::vector<ColumnType> types;
  for (const auto& column : columns) {
    types.push_back(column->getDataType());
  }
  return types;
}

ColumnBase* DataTable::getColumn(size_t index) const {
  if (index >= columns.size()) {
    throw std::out_of_range("Column index out of bounds in getColumn");
  }
  return columns[index].get();
}

int DataTable::getColumnIndex(const std::string& name) const {
  for (size_t i = 0; i < columns.size(); ++i) {
    if (columns[i]->name == name) {
      return i;
    }
  }
  return -1;
}

ColumnBase* DataTable::getColumnByName(const std::string& name) const {
  int index = getColumnIndex(name);
  return (index != -1) ? columns[index].get() : nullptr;
}

bool DataTable::hasColumn(const std::string& name) const { return getColumnIndex(name) != -1; }

void DataTable::addColumn(std::unique_ptr<ColumnBase> column) {
  if (column->length() != getNumRows()) {
    throw std::runtime_error("Column '" + column->name + "' has inconsistent number of rows: expected " +
                             std::to_string(getNumRows()) + ", got " + std::to_string(column->length()));
  }
  columns.push_back(std::move(column));
}

bool DataTable::removeColumn(const std::string& name) {
  auto it = std::remove_if(columns.begin(), columns.end(), [&name](const auto& col) { return col->name == name; });
  if (it == columns.end()) {
    return false;
  }
  columns.erase(it, columns.end());
  return true;
}

DataTable DataTable::clone() const {
  std::vector<std::unique_ptr<ColumnBase>> cloned_columns;
  for (const auto& column : columns) {
    cloned_columns.push_back(column->clone());
  }
  return DataTable(std::move(cloned_columns));
}

DataTable DataTable::permuteRows(const std::vector<uint32_t>& indices) const {
  std::vector<std::unique_ptr<ColumnBase>> new_columns;
  size_t new_length = indices.size();

  for (const auto& old_column_base : columns) {
    auto col_type = old_column_base->getDataType();
    std::unique_ptr<ColumnBase> new_col = createEmptyColumn(old_column_base->name, col_type, 0);

    new_col->permuteData(old_column_base.get(), indices);

    new_columns.push_back(std::move(new_col));
  }

  return DataTable(std::move(new_columns));
}

std::unique_ptr<ColumnBase> DataTable::createEmptyColumn(const std::string& name, ColumnType type,
                                                         size_t length) const {
  switch (type) {
    case ColumnType::INT8:
      return std::make_unique<Column<int8_t>>(name, std::vector<int8_t>(length));
    case ColumnType::UINT8:
      return std::make_unique<Column<uint8_t>>(name, std::vector<uint8_t>(length));
    case ColumnType::INT16:
      return std::make_unique<Column<int16_t>>(name, std::vector<int16_t>(length));
    case ColumnType::UINT16:
      return std::make_unique<Column<uint16_t>>(name, std::vector<uint16_t>(length));
    case ColumnType::INT32:
      return std::make_unique<Column<int32_t>>(name, std::vector<int32_t>(length));
    case ColumnType::UINT32:
      return std::make_unique<Column<uint32_t>>(name, std::vector<uint32_t>(length));
    case ColumnType::FLOAT32:
      return std::make_unique<Column<float>>(name, std::vector<float>(length));
    case ColumnType::FLOAT64:
      return std::make_unique<Column<double>>(name, std::vector<double>(length));
    default:
      throw std::runtime_error("Unsupported column type in factory.");
  }
}

/**
 * @brief Spreads the bits of a 10-bit integer using a magic bit sequence
 * (based on the method described by F. Giesen).
 * Used to encode coordinates into a Morton code.
 * @param x The 10-bit integer component (0-1023).
 * @return The spread integer (bits separated by two zeros).
 */
uint32_t Part1By2(uint32_t x) {
  // Restrict to 10 bits: x &= 0x000003ff;
  x &= 0x3ff;

  // x = (x ^ (x << 16)) & 0xff0000ff;
  x = (x ^ (x << 16)) & 0xff0000ff;

  // x = (x ^ (x << 8)) & 0x0300f00f;
  x = (x ^ (x << 8)) & 0x300f00f;

  // x = (x ^ (x << 4)) & 0x030c30c3;
  x = (x ^ (x << 4)) & 0x30c30c3;

  // x = (x ^ (x << 2)) & 0x09249249;
  x = (x ^ (x << 2)) & 0x9249249;

  return x;
}

/**
 * @brief Encodes a 3D coordinate (each component 0-1023) into a 30-bit Morton code.
 * @param x X-coordinate (0-1023).
 * @param y Y-coordinate (0-1023).
 * @param z Z-coordinate (0-1023).
 * @return The 30-bit Morton code (uint32_t).
 */
uint32_t encodeMorton3(uint32_t x, uint32_t y, uint32_t z) {
  // Interleave the bits: M = zzzzzzzzzz yyyyyyyyyy xxxxxxxxxx
  return (Part1By2(z) << 2) | (Part1By2(y) << 1) | Part1By2(x);
}

/**
 * @brief Generates a spatial ordering of point indices using 3D Morton codes
 * with recursive refinement for large buckets.
 * * @param dataTable The DataTable containing 'x', 'y', and 'z' coordinate columns.
 * @param indices A vector of indices (row numbers) to be sorted. MODIFIED IN PLACE.
 * @return The spatially sorted vector of indices (reference to the modified input).
 */
std::vector<uint32_t>& generateOrdering(DataTable& dataTable, std::vector<uint32_t>& indices) {
  if (indices.empty()) {
    return indices;
  }

  // Helper to safely retrieve coordinate values using the DataTable interface
  auto getVal = [&](const std::string& name, size_t index) -> double {
    ColumnBase* col = dataTable.getColumnByName(name);
    if (!col) {
      // Throw an exception if a required column is missing
      throw std::runtime_error("Required column '" + name + "' not found.");
    }
    return col->getValue(index);
  };

  // Define the recursive function using std::function
  // The indices vector passed to 'generate' represents the current sub-array to be sorted.
  std::function<void(std::vector<uint32_t>&)> generate;

  generate = [&](std::vector<uint32_t>& currentIndices) {
    if (currentIndices.empty()) {
      return;
    }

    double mx, my, mz;  // Minimum extent
    double Mx, My, Mz;  // Maximum extent

    // 1. Calculate scene extents across the current set of indices

    // Initialize extents with the first point
    mx = Mx = getVal("x", currentIndices[0]);
    my = My = getVal("y", currentIndices[0]);
    mz = Mz = getVal("z", currentIndices[0]);

    for (size_t i = 1; i < currentIndices.size(); ++i) {
      const size_t ri = currentIndices[i];  // Row index in the DataTable
      const double x = getVal("x", ri);
      const double y = getVal("y", ri);
      const double z = getVal("z", ri);

      if (x < mx)
        mx = x;
      else if (x > Mx)
        Mx = x;
      if (y < my)
        my = y;
      else if (y > My)
        My = y;
      if (z < mz)
        mz = z;
      else if (z > Mz)
        Mz = z;
    }

    const double xlen = Mx - mx;
    const double ylen = My - my;
    const double zlen = Mz - mz;

    // Check for invalid (non-finite) extents
    if (!std::isfinite(xlen) || !std::isfinite(ylen) || !std::isfinite(zlen)) {
      // logger.debug equivalent
      std::cerr << "WARNING: Invalid extents detected in generateOrdering.\n";
      return;
    }

    // All points are identical (zero extent)
    if (xlen == 0.0 && ylen == 0.0 && zlen == 0.0) {
      return;
    }

    // 2. Calculate scaling multipliers (to map extents to [0, 1024])
    const double MAX_MORTON_COORD = 1024.0;

    const double xmul = (xlen == 0.0) ? 0.0 : MAX_MORTON_COORD / xlen;
    const double ymul = (ylen == 0.0) ? 0.0 : MAX_MORTON_COORD / ylen;
    const double zmul = (zlen == 0.0) ? 0.0 : MAX_MORTON_COORD / zlen;

    // 3. Calculate Morton codes for all points in the current batch
    std::vector<uint32_t> morton(currentIndices.size());
    for (size_t i = 0; i < currentIndices.size(); ++i) {
      const size_t ri = currentIndices[i];
      const double x = getVal("x", ri);
      const double y = getVal("y", ri);
      const double z = getVal("z", ri);

      // Scale and clamp to [0, 1023] (integer space)
      uint32_t ix = static_cast<uint32_t>(std::min(1023.0, (x - mx) * xmul));
      uint32_t iy = static_cast<uint32_t>(std::min(1023.0, (y - my) * ymul));
      uint32_t iz = static_cast<uint32_t>(std::min(1023.0, (z - mz) * zmul));

      morton[i] = encodeMorton3(ix, iy, iz);
    }

    // 4. Create an Order array (0, 1, 2, ...) to sort by Morton code
    std::vector<uint32_t> order(currentIndices.size());
    std::iota(order.begin(), order.end(), 0);

    // Sort the 'order' array based on the corresponding 'morton' codes
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return morton[a] < morton[b]; });

    // 5. Apply the sorting to the 'currentIndices' vector (permute in place)
    // Since we are sorting a vector *in place*, we must copy the source indices first.
    std::vector<uint32_t> tmpIndices = currentIndices;
    for (size_t i = 0; i < currentIndices.size(); ++i) {
      currentIndices[i] = tmpIndices[order[i]];
    }

    // 6. Recursively sort the largest buckets (groups with identical Morton codes)
    size_t start = 0;
    size_t end = 1;
    const size_t BUCKET_THRESHOLD = 256;

    while (start < currentIndices.size()) {
      // Find the end of the current bucket
      while (end < currentIndices.size() && morton[order[end]] == morton[order[start]]) {
        ++end;
      }

      // If the bucket size is greater than the threshold, recurse
      if (end - start > BUCKET_THRESHOLD) {
        // Create a sub-vector (C++ equivalent of indices.subarray(start, end))
        std::vector<uint32_t> sub_indices(currentIndices.begin() + start, currentIndices.begin() + end);

        // Recursive call
        generate(sub_indices);

        // Copy the sorted sub_indices back into the main vector
        std::copy(sub_indices.begin(), sub_indices.end(), currentIndices.begin() + start);
      }

      start = end;
    }
  };

  // Initial call to the recursive sorting function
  generate(indices);

  return indices;
}

// Pre-define SH coefficient column names (f_rest_0 to f_rest_44)
const std::vector<std::string> shNames = []() {
  std::vector<std::string> names(45);
  for (int i = 0; i < 45; ++i) {
    names[i] = "f_rest_" + std::to_string(i);
  }
  return names;
}();

/**
 * @brief Applies translation, rotation, and scale to all Gaussian points in a DataTable.
 * * @param dataTable The DataTable containing the Gaussian data (positions, rotations, scales, SH).
 * @param t Global translation vector (Vec3).
 * @param r Global rotation quaternion (Quat).
 * @param s Global uniform scale factor (float).
 * @throws std::runtime_error if dataTable operations fail.
 */
void transform(DataTable& dataTable, const Eigen::Vector3d& t, const Eigen::Quaterniond& r, float s) {
  // 1. Pre-calculate global transformation matrices and SH rotation utility

  // Mat4: Global Transformation Matrix (TRS)
  // Eigen uses Column-Major by default. PlayCanvas's setTRS results in a matrix that,
  // when applied to a vector (v' = M * v), performs T * R * S.
  Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
  mat.block<3, 3>(0, 0) = r.toRotationMatrix() * s;  // R * S
  mat.block<3, 1>(0, 3) = t;                         // T (translation column)

  // Mat3: Pure Rotation Matrix for SH rotation
  Eigen::Matrix3d mat3 = r.toRotationMatrix();
  RotateSH rotateSH(mat3.cast<double>());  // Use double for Eigen's RotateSH as per assumed definition

  // 2. Determine which components exist in the DataTable (Optimization)

  const bool hasTranslation = dataTable.hasColumn("x") && dataTable.hasColumn("y") && dataTable.hasColumn("z");
  const bool hasRotation = dataTable.hasColumn("rot_0") && dataTable.hasColumn("rot_1") &&
                           dataTable.hasColumn("rot_2") && dataTable.hasColumn("rot_3");
  const bool hasScale =
      dataTable.hasColumn("scale_0") && dataTable.hasColumn("scale_1") && dataTable.hasColumn("scale_2");

  // Determine SH bands and coefficient count
  // The original logic finds the first missing f_rest_i column to infer the band count.
  // Index: 0-8 (9 coeffs, L0+L1), 0-23 (24 coeffs, L0-L2), 0-44 (45 coeffs, L0-L3)
  int missingIndex = -1;
  for (int i = 0; i < 45; ++i) {
    if (!dataTable.hasColumn(shNames[i])) {
      missingIndex = i;
      break;
    }
  }

  // Mapping from (first missing index) to (SH Band Index: 0=None, 1=L1, 2=L2, 3=L3)
  // 45 is total max, 9 is L0+L1, 24 is L0-L2.
  int shBands = 0;           // 0=None, 1=L1 (9 coeffs), 2=L2 (24 coeffs), 3=L3 (45 coeffs)
  if (missingIndex == -1) {  // All 45 present (L0-L3)
    shBands = 3;
  } else if (missingIndex <= 9) {  // Only L0 (9/3=3 coeffs) or L1 (9 coeffs)
    // NOTE: The original logic is ambiguous for L0 (1 coeff). Assuming if f_rest_0 to f_rest_8 are present, it is L1.
    if (missingIndex > 8) {  // f_rest_0 to f_rest_8 are present
      shBands = 1;
    } else if (missingIndex > 23) {  // f_rest_0 to f_rest_23 are present
      shBands = 2;
    }
  }

  // Number of coefficients per color channel (R, G, B) for the SH bands being rotated (excluding L0)
  // Total L0-L3 coeffs: 1, 3, 8, 15 (per channel)
  // L0: 1 (L0 only)
  // L1: 3 (L1 only)
  // L2: 8 (L2 only)
  // L3: 15 (L3 only)

  // The coefficients to be rotated start from L1 (index 1) for a total of 3, 8, or 15 coeffs.
  // Total coeffs are 1 (L0) + 3*L1 + 5*L2 + 7*L3...
  // The `f_rest_i` columns hold the L1+ coefficients (total: 3* (n^2 - 1) coeffs)
  // The original JS uses 3, 8, 15 which corresponds to L1, L2, L3 bands *excluding L0*.
  // The array `shCoeffs` holds the coefficients *per color channel* (L1, L2, L3).
  const int shCoeffsPerChannel = (shBands == 1) ? 3 : (shBands == 2) ? 8 : (shBands == 3) ? 15 : 0;

  if (shBands > 0) {
    std::cout << "Applying SH rotation with " << shBands << " band(s) (" << shCoeffsPerChannel
              << " coeffs per channel)." << std::endl;
  }

  // Temporary buffer for SH coefficients of one color channel
  std::vector<double> shCoeffs(shCoeffsPerChannel);

  // 3. Iterate and Transform Rows
  for (size_t i = 0; i < dataTable.getNumRows(); ++i) {
    // Use a temporary map to hold row data for read/write
    Row row = dataTable.getRow(i);

    // --- A. Translation (Position) ---
    if (hasTranslation) {
      Eigen::Vector3d pos(row["x"], row["y"], row["z"]);

      // Transform point: v' = M * v
      // Since Eigen's Mat4 * Vec3 (implicitly converting Vec3 to Vec4(v, 1))
      // already performs the correct projective transform (v' = T*R*S*v), we use that.
      Eigen::Vector4f pos4(pos.x(), pos.y(), pos.z(), 1.0f);
      pos4 = mat * pos4;

      row["x"]= pos4.x() / pos4.w();
      row["y"]= pos4.y() / pos4.w();
      row["z"]= pos4.z() / pos4.w();
    }

    // --- B. Rotation ---
    if (hasRotation) {
      // Note: The original code uses (rot_1, rot_2, rot_3, rot_0) -> (x, y, z, w) Playcanvas convention,
      // then multiplies by the global rotation 'r' (Quat).
      // Original: q.set(row.rot_1, row.rot_2, row.rot_3, row.rot_0).mul2(r, q);
      // Eigen stores Quat as (x, y, z, w) in memory/coefficients.
      Eigen::Quaterniond q_local(row["rot_0"],  // w
                                 row["rot_1"],  // x
                                 row["rot_2"],  // y
                                 row["rot_3"]   // z
      );

      // The combined rotation: q_global * q_local
      Eigen::Quaterniond q_combined = r * q_local;

      // Re-normalize might be a good idea, though quaternion multiplication should maintain unit length
      q_combined.normalize();

      // Store back using the original (w, x, y, z) column convention
      row["rot_0"] = q_combined.w();
      row["rot_1"] = q_combined.x();
      row["rot_2"] = q_combined.y();
      row["rot_3"] = q_combined.z();
    }

    // --- C. Scale ---
    if (hasScale) {
      // Scale is stored as log(scale). Scale transformation: exp(log(s_old)) * s_global = s_new.
      // log(s_new) = log(exp(log(s_old)) * s_global) = log(s_old) + log(s_global)
      float log_s = std::log(s);

      row["scale_0"] = row["scale_0"] + log_s;
      row["scale_1"] = row["scale_1"] + log_s;
      row["scale_2"] = row["scale_2"] + log_s;
    }

    // --- D. Spherical Harmonics (SH) Rotation ---
    if (shBands > 0) {
      // Iterate over the three color channels (R, G, B)
      for (int j = 0; j < 3; ++j) {
        // 1. Load SH coefficients for one channel (R, G, or B)
        for (int k = 0; k < shCoeffsPerChannel; ++k) {
          const std::string& colName = shNames[k + j * shCoeffsPerChannel];
          shCoeffs[k] = row[colName];
        }

        // 2. Apply rotation (RotateSH is assumed to operate on the double vector)
        rotateSH.apply(shCoeffs);

        // 3. Store rotated SH coefficients back to the row
        for (int k = 0; k < shCoeffsPerChannel; ++k) {
          const std::string& colName = shNames[k + j * shCoeffsPerChannel];
          row[colName] = shCoeffs[k];
        }
      }
    }

    // --- E. Final Update ---
    dataTable.setRow(i, row);
  }
}

}  // namespace splat
