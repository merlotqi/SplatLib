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

#include <splat/io/ply_writer.h>

#include <fstream>
#include <numeric>
#include <vector>

namespace splat {

static std::string columnTypeToPlyType(ColumnType type) {
  switch (type) {
    case ColumnType::INT8:
      return "char";
    case ColumnType::UINT8:
      return "uchar";
    case ColumnType::INT16:
      return "short";
    case ColumnType::UINT16:
      return "ushort";
    case ColumnType::INT32:
      return "int";
    case ColumnType::UINT32:
      return "uint";
    case ColumnType::FLOAT32:
      return "float";
    case ColumnType::FLOAT64:
      return "double";
    default:
      throw std::runtime_error("Unsupported column type");
  }
}

void writePly(const std::string& filename, const PlyData& plyData) {
  // header strings
  std::vector<std::string> header;
  header.emplace_back("ply");
  header.emplace_back("format binary_little_endian 1.0");
  for (auto&& c : plyData.comments) {
    header.emplace_back("comment " + c);
  }
  for (auto&& element : plyData.elements) {
    header.emplace_back("element " + element.name + " " + std::to_string(element.dataTable->getNumRows()));
    for (auto&& column : element.dataTable->columns) {
      header.emplace_back("property " + columnTypeToPlyType(column.getType()) + " " + column.name);
    }
  }
  header.emplace_back("end_header");

  std::ofstream ofs(filename, std::ios::binary | std::ios::out);
  if (!ofs.is_open()) {
    throw std::runtime_error("Could not open file for writing");
  }

  // write the header
  for (const auto& h : header) {
    ofs << h << "\n";
  }
  ofs.flush();

  for (size_t i = 0; i < plyData.elements.size(); ++i) {
    const auto& table = plyData.elements[i].dataTable;
    const auto& columns = table->columns;
    std::vector<const uint8_t*> buffers;
    std::vector<size_t> sizes;
    for (const auto& column : columns) {
      sizes.push_back(column.bytePreElement());
      buffers.push_back(column.rawPointer());
    }
    std::size_t rowSize = std::reduce(sizes.begin(), sizes.end(), size_t{0});

    // write to file in chunks of 1024 rows
    const size_t chunkSize = 1024;
    const size_t numChunks = ceil(static_cast<double>(table->getNumRows()) / chunkSize);
    std::vector<uint8_t> chunkData(chunkSize * rowSize);

    for (size_t c = 0; c < numChunks; ++c) {
      const auto numRows = std::min(chunkSize, table->getNumRows() - c * chunkSize);

      size_t offset = 0;

      for (size_t r = 0; r < numRows; ++r) {
        const auto rowOffset = c * chunkSize + r;

        for (size_t p = 0; p < columns.size(); ++p) {
          const auto s = sizes[p];
          memcpy(chunkData.data() + offset, buffers[p] + rowOffset * s, s);
          offset += s;
        }
      }

      // write the chunk
      ofs.write(reinterpret_cast<const char*>(chunkData.data()), numRows * rowSize);
      ofs.flush();
    }
  }
  ofs.close();
}

}  // namespace splat
