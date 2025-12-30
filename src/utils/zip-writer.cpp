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

#include <splat/utils/zip-writer.h>

#include <chrono>

namespace splat {

namespace ZipConstants {

static constexpr uint32_t SIG_LFH = 0x04034b50;
static constexpr uint32_t SIG_CDR = 0x02014b50;
static constexpr uint32_t SIG_EOCD = 0x06054b50;
static constexpr uint32_t SIG_DESCRIPTOR = 0x08074b50;
static constexpr uint16_t LFH_FIXED_SIZE = 30;
static constexpr uint16_t CDR_FIXED_SIZE = 46;

}  // namespace ZipConstants

/**
 * @brief Writes a 16-bit unsigned integer to the stream in Little-Endian byte order.
 */
void ZipWriter::writeUint16LE(uint16_t value) { file_.write(reinterpret_cast<const char*>(&value), 2); }

/**
 * @brief Writes a 32-bit unsigned integer to the stream in Little-Endian byte order.
 */
void ZipWriter::writeUint32LE(uint32_t value) { file_.write(reinterpret_cast<const char*>(&value), 4); }

/**
 * @brief Writes the 16-byte Data Descriptor for the last finished file.
 */
void ZipWriter::writeDataDescriptor() {
  if (files_.empty()) return;

  const FileInfo& fileInfo = files_.back();

  // 1. Signature (0x08074b50)
  writeUint32LE(ZipConstants::SIG_DESCRIPTOR);

  // 2. CRC-32 value
  writeUint32LE(fileInfo.crc.value());

  // 3. Compressed Size (equal to uncompressed size for STORE method)
  writeUint32LE(fileInfo.sizeBytes);

  // 4. Uncompressed Size
  writeUint32LE(fileInfo.sizeBytes);
}

/**
 * @brief Writes the Local File Header (LFH) for a new file and updates metadata.
 * Captures the offset before writing the header.
 * @param filenameBuf The UTF-8 encoded filename bytes.
 */
void ZipWriter::writeLocalFileHeader(const std::vector<uint8_t>& filenameBuf) {
  const uint16_t nameLen = static_cast<uint16_t>(filenameBuf.size());

  // Capture the offset of the LFH before writing it
  uint32_t currentOffset = static_cast<uint32_t>(file_.tellp());

  // --- Write LFH Fixed Part (30 bytes) ---

  // 1. Signature (0x04034b50)
  writeUint32LE(ZipConstants::SIG_LFH);

  // 2. Version needed to extract (2.0)
  writeUint16LE(20);

  // 3. General Purpose Flag: 0x8 (Data descriptor used) | 0x800 (UTF-8 encoding)
  writeUint16LE(0x8 | 0x800);

  // 4. Compression Method (0 = STORED)
  writeUint16LE(0);

  // 5. Last Modified Time/Date
  writeUint16LE(dosTime_);
  writeUint16LE(dosDate_);

  // 6. CRC-32, Compressed Size, Uncompressed Size (all zero since data descriptor will follow)
  writeUint32LE(0);
  writeUint32LE(0);
  writeUint32LE(0);

  // 7. Filename Length, Extra Field Length (0)
  writeUint16LE(nameLen);
  writeUint16LE(0);

  // 8. Filename
  file_.write(reinterpret_cast<const char*>(filenameBuf.data()), nameLen);

  // Store metadata for the Central Directory Record
  FileInfo info;
  info.localHeaderOffset = currentOffset;
  info.filenameBuf = filenameBuf;
  files_.emplace_back(info);
}

/**
 * @brief Construct a new Zip Writer object and open the output file.
 * @param filename Path to the output ZIP file.
 * @throws std::runtime_error if the file cannot be opened.
 */
ZipWriter::ZipWriter(const std::string& filename) {
  file_.open(filename, std::ios::binary | std::ios::out);
  if (!file_.is_open()) {
    throw std::runtime_error("Failed to open output file: " + filename);
  }

  // Calculate DOS date and time fields based on current time
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  // NOTE: std::localtime is not thread-safe, but used here for simplicity.
  std::tm* tm = std::localtime(&t);

  dosTime_ = static_cast<uint16_t>((tm->tm_hour << 11) | (tm->tm_min << 5) | (tm->tm_sec / 2));
  dosDate_ = static_cast<uint16_t>(((tm->tm_year + 1900 - 1980) << 9) | ((tm->tm_mon + 1) << 5) | tm->tm_mday);
}

/**
 * @brief Starts a new file entry in the archive. Writes the LFH.
 * Automatically writes the Data Descriptor for the previous file, if one exists.
 * @param filename The name of the file to be added.
 * @throws std::runtime_error if file stream is not open.
 */
void ZipWriter::start(const std::string& filename) {
  if (!file_.is_open()) {
    throw std::runtime_error("File stream is not open.");
  }

  // Write descriptor for previous file
  if (!files_.empty()) {
    writeDataDescriptor();
  }

  // Convert filename to UTF-8 bytes and write new LFH
  std::vector<uint8_t> filenameBuf(filename.begin(), filename.end());
  writeLocalFileHeader(filenameBuf);
}

/**
 * @brief Writes a chunk of raw data to the currently open file entry.
 * Updates CRC and size counters.
 * @param data Pointer to the raw byte data.
 * @param length The length of the data chunk.
 * @throws std::runtime_error if no file has been started.
 */
void ZipWriter::write(const uint8_t* data, size_t length) {
  if (files_.empty()) {
    throw std::runtime_error("Cannot write data: must call start() first.");
  }

  FileInfo& currentFile = files_.back();

  // Update metadata
  currentFile.sizeBytes += static_cast<uint32_t>(length);
  currentFile.crc.update(data, length);  // Assumes Crc::update overload for raw pointer/length

  // Write data to stream
  file_.write(reinterpret_cast<const char*>(data), length);
}

/**
 * @brief Overload to write data from a vector.
 */
void ZipWriter::write(const std::vector<uint8_t>& data) { write(data.data(), data.size()); }

/**
 * @brief Finishes the archive by writing the final data descriptor,
 * the Central Directory records, and the EOCD.
 * Flushes and closes the underlying file stream.
 */
void ZipWriter::close() {
  if (!file_.is_open()) return;

  // 1. Write Data Descriptor for the last file
  if (!files_.empty()) {
    writeDataDescriptor();
  }

  // Record the start position of the Central Directory
  uint32_t centralDirOffset = static_cast<uint32_t>(file_.tellp());
  uint32_t centralDirSize = 0;

  // 2. Write Central Directory (CD) records
  for (const auto& fileInfo : files_) {
    const uint16_t nameLen = static_cast<uint16_t>(fileInfo.filenameBuf.size());

    // --- Write CD Record Fixed Part (46 bytes) ---

    // 1. Signature (0x02014b50)
    writeUint32LE(ZipConstants::SIG_CDR);

    // 2. Version made by (2.0) / 3. Version needed to extract (2.0)
    writeUint16LE(20);
    writeUint16LE(20);

    // 4. General Purpose Flag (0x8 | 0x800)
    writeUint16LE(0x8 | 0x800);

    // 5. Compression Method (0)
    writeUint16LE(0);

    // 6. Last Modified Time/Date
    writeUint16LE(dosTime_);
    writeUint16LE(dosDate_);

    // 7. CRC-32 / 8. Compressed Size / Uncompressed Size
    writeUint32LE(fileInfo.crc.value());
    writeUint32LE(fileInfo.sizeBytes);
    writeUint32LE(fileInfo.sizeBytes);

    // 9. Filename Length / Extra Field Length (0) / File Comment Length (0)
    writeUint16LE(nameLen);
    writeUint16LE(0);
    writeUint16LE(0);

    // 10. Disk Number (0) / Internal Attributes (0) / External Attributes (0)
    writeUint16LE(0);
    writeUint16LE(0);
    writeUint32LE(0);

    // 11. Offset of Local File Header
    writeUint32LE(fileInfo.localHeaderOffset);

    // 12. Filename
    file_.write(reinterpret_cast<const char*>(fileInfo.filenameBuf.data()), nameLen);

    centralDirSize += ZipConstants::CDR_FIXED_SIZE + nameLen;
  }

  // 3. Write End of Central Directory Record (EOCD)

  // 1. Signature (0x06054b50)
  writeUint32LE(ZipConstants::SIG_EOCD);

  // 2. Disk number / Disk number start of CD (0)
  writeUint16LE(0);
  writeUint16LE(0);

  // 3. Number of entries in this disk / Total number of entries
  uint16_t fileCount = static_cast<uint16_t>(files_.size());
  writeUint16LE(fileCount);
  writeUint16LE(fileCount);

  // 4. Size of Central Directory
  writeUint32LE(centralDirSize);

  // 5. Offset of start of Central Directory relative to start of archive
  writeUint32LE(centralDirOffset);

  // 6. Comment length (0)
  writeUint16LE(0);

  // 4. Close the file stream
  file_.close();
}

// --- Convenience Overloads for File Writing ---

void ZipWriter::writeFile(const std::string& filename, const std::string& content) {
  start(filename);
  write(reinterpret_cast<const uint8_t*>(content.data()), content.size());
}

void ZipWriter::writeFile(const std::string& filename, const std::vector<uint8_t>& content) {
  start(filename);
  write(content.data(), content.size());
}

void ZipWriter::writeFile(const std::string& filename, const std::vector<std::vector<uint8_t>>& content) {
  start(filename);
  for (const auto& chunk : content) {
    write(chunk.data(), chunk.size());
  }
}

}  // namespace splat
