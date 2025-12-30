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

#include <splat/utils/crc.h>

#include <cstdint>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>

namespace splat {

/**
 * @brief Represents the metadata for a single file entry within the archive.
 * Used to construct the Central Directory Record (CDR).
 */
struct FileInfo {
  std::vector<uint8_t> filenameBuf;  // UTF-8 encoded filename bytes
  Crc crc;                           // CRC-32 checksum calculator
  uint32_t sizeBytes = 0;            // Uncompressed size (for STORE method)
  uint32_t localHeaderOffset = 0;    // Byte offset of the Local File Header (LFH)
};

/**
 * @brief Synchronous streaming ZIP archive writer.
 * Handles ZIP format encoding for uncompressed (STORED) files using data descriptors.
 * It manages the output file stream internally (RAII principle).
 */
class ZipWriter {
 private:
  std::ofstream file_;  // The internal file stream (owned by ZipWriter)
  std::vector<FileInfo> files_;

  // DOS Date/Time fields, calculated once upon initialization
  uint16_t dosTime_ = 0;
  uint16_t dosDate_ = 0;

  void writeUint16LE(uint16_t value);
  void writeUint32LE(uint32_t value);
  void writeDataDescriptor();
  void writeLocalFileHeader(const std::vector<uint8_t>& filenameBuf);

 public:
  explicit ZipWriter(const std::string& filename);

  ZipWriter(ZipWriter&&) noexcept = default;
  ZipWriter& operator=(ZipWriter&&) noexcept = default;
  ZipWriter(const ZipWriter&) = delete;
  ZipWriter& operator=(const ZipWriter&) = delete;

  void start(const std::string& filename);
  void write(const uint8_t* data, size_t length);
  void write(const std::vector<uint8_t>& data);
  void close();

  void writeFile(const std::string& filename, const std::string& content);
  void writeFile(const std::string& filename, const std::vector<uint8_t>& content);
  void writeFile(const std::string& filename, const std::vector<std::vector<uint8_t>>& content);
};

}  // namespace splat
