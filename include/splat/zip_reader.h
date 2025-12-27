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

#include <cstdint>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace splat {

class ZipEntry {
 public:
  std::string name;
  uint32_t size;                                   // Uncompressed size
  std::function<std::vector<uint8_t>()> readData;  // Lazy data read function
  ZipEntry(std::string n, uint32_t s, std::function<std::vector<uint8_t>()> rd)
      : name(std::move(n)), size(s), readData(std::move(rd)) {}
};

/**
 * @brief Minimal ZIP reader supporting STORED (method 0) and data descriptors.
 * It sequentially parses Local File Headers to list entries.
 */
class ZipReader {
 private:
  std::ifstream file;
  // Use std::streamoff for offsets; it's designed for signed differences/offsets,
  // though std::streampos is acceptable for absolute positions.
  std::streamoff cursor = 0;
  std::streamoff fileSize = 0;

 public:
  explicit ZipReader(const std::string& filename);
  ~ZipReader();

  ZipReader(const ZipReader&) = delete;
  ZipReader& operator=(const ZipReader&) = delete;

  ZipReader(ZipReader&& other) noexcept = default;
  ZipReader& operator=(ZipReader&& other) noexcept = default;

  std::vector<ZipEntry> list();

 private:
  std::vector<uint8_t> readAt(std::streamoff pos, size_t len);
  std::vector<uint8_t> read(size_t len);
  uint32_t readUint32LE(const std::vector<uint8_t>& data, size_t offset);
  uint16_t readUint16LE(const std::vector<uint8_t>& data, size_t offset);
  std::string decodeName(const std::vector<uint8_t>& nameBytes, bool utf8);
};

}  // namespace splat
