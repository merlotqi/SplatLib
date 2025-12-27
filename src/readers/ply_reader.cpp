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

#include <splat/readers/ply_reader.h>

#include <algorithm>
#include <fstream>
#include <numeric>

namespace splat {

static const std::vector<uint8_t> magicBytes = {'p', 'l', 'y', '\n'};
static const std::vector<uint8_t> endHeaderBytes = {'\n', 'e', 'n', 'd', '_', 'h', 'e', 'a', 'd', 'e', 'r', '\n'};

static bool cmp(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, size_t aOffset = 0) {
  if (aOffset + b.size() > a.size()) {
    return false;
  }

  return std::equal(b.begin(), b.end(), a.begin() + aOffset);
}

static Column createColumn(const std::string& name, ColumnType type, size_t count) {
  switch (type) {
    case ColumnType::INT8:
      return {name, std::vector<uint8_t>(count)};
    case ColumnType::UINT8:
      return {name, std::vector<uint8_t>(count)};
    case ColumnType::INT16:
      return {name, std::vector<uint16_t>(count)};
    case ColumnType::UINT16:
      return {name, std::vector<uint16_t>(count)};
    case ColumnType::INT32:
      return {name, std::vector<int32_t>(count)};
    case ColumnType::UINT32:
      return {name, std::vector<uint32_t>(count)};
    case ColumnType::FLOAT32:
      return {name, std::vector<float>(count)};
    case ColumnType::FLOAT64:
      return {name, std::vector<double>(count)};
  }
  return {};
}

/**
 * @brief Maps PLY header type strings to C++ ColumnType and byte size.
 */
static ColumnType getDataTypeMapping(const std::string& type) {
  static const std::map<std::string, ColumnType> typeMap = {
      {"char", ColumnType::INT8},       {"uchar", ColumnType::UINT8},    {"short", ColumnType::INT16},
      {"ushort", ColumnType::UINT16},   {"int", ColumnType::INT32},      {"uint", ColumnType::UINT32},
      {"float", ColumnType::FLOAT32},   {"double", ColumnType::FLOAT64}, {"float32", ColumnType::FLOAT32},
      {"float64", ColumnType::FLOAT64},
  };
  auto it = typeMap.find(type);
  if (it != typeMap.end()) {
    return it->second;
  }
  throw std::runtime_error("Unsupported PLY data type: " + type);
}

/**
 * @brief Parses the PLY header text data into structured PlyHeader components.
 * @param data Buffer containing the PLY header text.
 * @return PlyHeader The parsed header information.
 */
static PlyHeader parseHeader(const std::vector<uint8_t>& data) {
  // Decode header and split into lines
  std::string headerStr(reinterpret_cast<const char*>(data.data()), data.size());
  std::stringstream ss(headerStr);
  std::string line;

  PlyHeader header;
  PlyElement* currentElement = nullptr;

  // skip the first line ('ply')
  std::getline(ss, line);

  while (std::getline(ss, line)) {
    // Remove trailing carriage return if present (for cross-platform compatibility)
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) continue;

    std::stringstream line_ss(line);
    std::string keyword;
    line_ss >> keyword;

    if (keyword == "ply" || keyword == "format" || keyword == "end_header") {
      // skip
    } else if (keyword == "comment") {
      // Extract the rest of the line after "comment "
      header.comments.push_back(line.substr(8));
    } else if (keyword == "element") {
      std::string name;
      size_t count;
      if (!(line_ss >> name >> count)) {
        throw std::runtime_error("invalid ply header: 'element' syntax error.");
      }
      header.elements.emplace_back(PlyElement{name, count, {}});
      currentElement = &header.elements.back();
    } else if (keyword == "property") {
      if (!currentElement) {
        throw std::runtime_error("invalid ply header: 'property' outside 'element'.");
      }
      std::string typeStr;
      std::string name;

      if (!(line_ss >> typeStr >> name)) {
        throw std::runtime_error("invalid ply header: 'property' syntax error.");
      }

      currentElement->properties.push_back({name, typeStr, getDataTypeMapping(typeStr)});
    } else {
      throw std::runtime_error("unrecognized header value '" + keyword + "' in ply header");
    }
  }
  return header;
}

std::unique_ptr<PlyData> readPly(const std::string& filename) {
  // open the file for binary input
  std::ifstream file(filename, std::ios::binary | std::ios::in);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  // read header --
  const size_t maxHeaderSize = 128 * 1024;
  std::vector<uint8_t> headerBuf(maxHeaderSize);
  if (static_cast<size_t>(file.read(reinterpret_cast<char*>(headerBuf.data()), magicBytes.size()).gcount()) !=
      magicBytes.size()) {
    throw std::runtime_error("Failed to read file header or file is too short.");
  }
  if (!cmp(headerBuf, magicBytes)) {
    throw std::runtime_error("Invalid file header: missing 'ply'.");
  }

  size_t headerSize = magicBytes.size();
  // Read the rest of the header until 'end_header' pattern is found
  while (headerSize < maxHeaderSize) {
    // read the next character
    if (file.read(reinterpret_cast<char*>(headerBuf.data() + headerSize), 1).gcount() != 1) {
      throw std::runtime_error("Failed to read file header: unexpected EOF.");
    }
    headerSize++;

    // Check for the 'end_header' byte pattern
    if (headerSize >= endHeaderBytes.size() && cmp(headerBuf, endHeaderBytes, headerSize - endHeaderBytes.size())) {
      break;
    }
  }

  if (headerSize >= maxHeaderSize) {
    throw std::runtime_error("PLY header too large (>128KB) or missing 'end_header'.");
  }

  // parse header --
  const std::vector<uint8_t> actualHeader(headerBuf.begin(), headerBuf.begin() + headerSize);
  PlyHeader header = parseHeader(actualHeader);

  // parse data --
  std::vector<PlyElementData> elements;
  for (auto& element : header.elements) {
    std::vector<Column> columns;
    std::vector<int> sizes;
    std::vector<uint8_t*> buffers;
    for (auto&& prop : element.properties) {
      columns.emplace_back(createColumn(prop.name, prop.dataType, element.count));
      sizes.emplace_back(columns.back().bytePreElement());
      buffers.emplace_back(columns.back().rawPointer());
    }

    size_t rowSize = std::reduce(sizes.begin(), sizes.end(), 0);

    // read data in chunks of 1024 rows at a time
    const size_t chunkSize = 1024;
    const size_t numChunks = ceil(element.count / chunkSize);
    std::vector<uint8_t> chunkData(chunkSize * rowSize);

    for (size_t c = 0; c < numChunks; ++c) {
      const size_t numRows = std::min(chunkSize, element.count - c * chunkSize);

      if (!file.read(reinterpret_cast<char*>(chunkData.data()), rowSize * numRows)) {
        throw std::runtime_error("Failed to read data chunk.");
      }

      size_t offset = 0;

      // read data row at a time
      for (size_t r = 0; r < numRows; ++r) {
        const auto rowOffset = c * chunkSize + r;

        // copy into column data
        for (size_t p = 0; p < columns.size(); ++p) {
          const auto s = sizes[p];
          std::memcpy(buffers[p] + rowOffset * s, chunkData.data() + offset, s);
          offset += s;
        }
      }
    }
    elements.push_back({element.name, std::make_unique<DataTable>(columns)});
  }

  auto result = std::make_unique<PlyData>();
  result->comments = std::move(header.comments);
  result->elements = std::move(elements);
  return result;
}

}  // namespace splat
