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

#include <splat/utils/zip-reader.h>

namespace splat {
namespace ZipConstants {

constexpr uint32_t LOCAL_FILE_HEADER_SIG = 0x04034b50;
constexpr uint32_t EOCD_SIG = 0x06054b50;
constexpr uint32_t CENTRAL_DIR_SIG = 0x02014b50;
constexpr uint32_t DATA_DESCRIPTOR_SIG = 0x08074b50;
constexpr uint16_t GP_FLAG_DATA_DESCRIPTOR = 0x8;  // Bit 3: Size/CRC are zeroed, followed by data descriptor
constexpr uint16_t GP_FLAG_UTF8 = 0x800;           // Bit 11: Filename is UTF-8 encoded
constexpr uint16_t METHOD_STORED = 0;              // Only STORED method (no compression) is supported
constexpr std::streamoff LFH_FIXED_SIZE = 30;              // Fixed size of Local File Header

}  // namespace ZipConstants

/**
 * @brief Synchronously reads 'len' bytes starting at absolute file position 'pos'.
 * @param pos The absolute file offset to start reading from.
 * @param len The number of bytes to read.
 * @return A vector containing the read bytes.
 * @throws std::runtime_error if EOF is reached unexpectedly.
 */
std::vector<uint8_t> ZipReader::readAt(std::streamoff pos, size_t len) {
  std::vector<uint8_t> buffer(len);
  file.seekg(pos);

  // Check if seek failed (e.g., pos is out of bounds)
  if (file.fail()) {
    file.clear();  // Clear the fail flag
    throw std::runtime_error("File seek failed to position: " + std::to_string(pos));
  }

  file.read(reinterpret_cast<char*>(buffer.data()), len);

  // Check if the expected number of bytes were read
  if (file.gcount() != static_cast<std::streamsize>(len)) {
    // Check for actual EOF before expected read length
    if (file.eof()) {
      throw std::runtime_error("Unexpected EOF while reading ZIP data.");
    }
    // Other read error
    throw std::runtime_error("File read failed, read " + std::to_string(file.gcount()) + " bytes, expected " +
                             std::to_string(len) + ".");
  }
  return buffer;
}

/**
 * @brief Synchronously reads 'len' bytes from the current cursor position.
 * Updates the cursor position.
 * @param len The number of bytes to read.
 * @return A vector containing the read bytes.
 */
std::vector<uint8_t> ZipReader::read(size_t len) {
  auto result = readAt(cursor, len);
  cursor += static_cast<std::streamoff>(len);
  return result;
}

// --- Byte-Order Utilities (Little Endian) ---

/**
 * @brief Reads a little-endian unsigned 32-bit integer from the buffer at the given offset.
 */
uint32_t ZipReader::readUint32LE(const std::vector<uint8_t>& data, size_t offset) {
  // Bounds checking is implicitly done by readAt/read if buffer creation failed.
  // Direct access assumes the buffer is correctly sized (30 bytes for header).
  return static_cast<uint32_t>(data[offset]) | (static_cast<uint32_t>(data[offset + 1]) << 8) |
         (static_cast<uint32_t>(data[offset + 2]) << 16) | (static_cast<uint32_t>(data[offset + 3]) << 24);
}

/**
 * @brief Reads a little-endian unsigned 16-bit integer from the buffer at the given offset.
 */
uint16_t ZipReader::readUint16LE(const std::vector<uint8_t>& data, size_t offset) {
  return static_cast<uint16_t>(data[offset]) | (static_cast<uint16_t>(data[offset + 1]) << 8);
}

/**
 * @brief Decodes the entry name from bytes based on the UTF-8 flag.
 * The simple conversion works for both UTF-8 and legacy/ASCII encoding.
 * @param nameBytes The raw byte vector of the filename.
 * @param utf8 True if the General Purpose Flag indicates UTF-8 encoding.
 * @return The decoded filename string.
 */
std::string ZipReader::decodeName(const std::vector<uint8_t>& nameBytes, bool utf8) {
  // Filename decoding is simplified. In C++, std::string construction from
  // raw bytes handles both UTF-8 and ASCII (single-byte character sets).
  // For non-UTF8/legacy encoding, OS locale interpretation applies,
  // which matches the behavior of the original TS logic.
  (void)utf8;
  return std::string(reinterpret_cast<const char*>(nameBytes.data()), nameBytes.size());
}

/**
 * @brief Construct a new Zip Reader object, opening the file and determining its size.
 * @param filename Path to the ZIP file.
 * @throws std::runtime_error if the file cannot be opened.
 */
ZipReader::ZipReader(const std::string& filename) {
  file.open(filename, std::ios::binary | std::ios::in);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  // Determine file size
  file.seekg(0, std::ios::end);
  fileSize = file.tellg();

  // Check if file size retrieval failed or is zero
  if (fileSize <= 0) {
    file.close();
    throw std::runtime_error("Failed to determine a valid file size for: " + filename);
  }

  file.seekg(0, std::ios::beg);
  cursor = 0;
}

/**
 * @brief Destructor ensures the file stream is closed.
 */
ZipReader::~ZipReader() {
  if (file.is_open()) {
    file.close();
  }
}

/**
 * @brief Synchronously lists all entries in the ZIP file by sequentially
 * parsing Local File Headers.
 * @return A vector of ZipEntry objects.
 */
std::vector<ZipEntry> ZipReader::list() {
  std::vector<ZipEntry> entries;
  cursor = 0;

  while (cursor + ZipConstants::LFH_FIXED_SIZE <= fileSize) {
    // 1. Read Local File Header (LFH) fixed part
    auto header = read(ZipConstants::LFH_FIXED_SIZE);
    uint32_t signature = readUint32LE(header, 0);

    // Check if Central Directory or EOCD signature is encountered (parsing stops)
    if (signature == ZipConstants::CENTRAL_DIR_SIG || signature == ZipConstants::EOCD_SIG) {
      // Backtrack cursor to the start of the signature
      cursor -= ZipConstants::LFH_FIXED_SIZE;
      break;
    }

    // Check for valid LFH signature
    if (signature != ZipConstants::LOCAL_FILE_HEADER_SIG) {
      cursor -= ZipConstants::LFH_FIXED_SIZE;
      break;
    }

    // 2. Parse LFH fields
    uint16_t gpFlags = readUint16LE(header, 6);
    uint16_t method = readUint16LE(header, 8);
    uint16_t nameLen = readUint16LE(header, 26);
    uint16_t extraLen = readUint16LE(header, 28);
    uint32_t size_header = readUint32LE(header, 22);
    readUint32LE(header, 14);

    // 3. Read name and extra fields, advancing the cursor
    auto nameBytes = read(nameLen);
    /* auto extra = */ read(extraLen);  // Extra field is read and discarded

    // 4. Decode filename and check method
    bool utf8 = (gpFlags & ZipConstants::GP_FLAG_UTF8) != 0;
    std::string name = decodeName(nameBytes, utf8);

    if (method != ZipConstants::METHOD_STORED) {
      throw std::runtime_error("Unsupported ZIP compression method: " + std::to_string(method) +
                               " (only STORE=0 supported)");
    }

    bool useDescriptor = (gpFlags & ZipConstants::GP_FLAG_DATA_DESCRIPTOR) != 0;
    std::streamoff dataOffset = cursor;
    uint32_t size = size_header;

    if (!useDescriptor) {
      // Case 1: Size is known in LFH (most common)
      std::streamoff start = dataOffset;
      size_t entrySize = size;
      cursor += static_cast<std::streamoff>(entrySize);

      // Create entry with lazy readData function
      auto readFunc = [this, start, entrySize]() -> std::vector<uint8_t> { return this->readAt(start, entrySize); };

      entries.emplace_back(name, entrySize, readFunc);
    } else {
      // Case 2: Data descriptor follows the file data (scan required)
      const size_t CHUNK_SIZE = 64 * 1024;
      std::streamoff pos = dataOffset;
      bool found = false;

      // Data descriptor signature bytes
      const uint32_t DESCRIPTOR_SIG = ZipConstants::DATA_DESCRIPTOR_SIG;

      while (pos < fileSize) {
        size_t toRead = static_cast<size_t>(std::min(static_cast<std::streamoff>(CHUNK_SIZE), fileSize - pos));

        // Optimization: Only read if there's enough space left for a descriptor search
        if (toRead < 16) break;

        auto buf = readAt(pos, toRead);

        // Search for signature in this chunk
        for (size_t i = 0; i + 16 <= buf.size(); i++) {
          if (readUint32LE(buf, i) == DESCRIPTOR_SIG) {
            // Found descriptor: read final size and advance cursor
            size = readUint32LE(buf, i + 8);
            // crc = readUint32LE(buf, i + 4); // CRC check is optional here

            std::streamoff endOfData = pos + static_cast<std::streamoff>(i);
            cursor = endOfData + 16;  // 16 bytes is the size of the descriptor

            // Create entry
            size_t entrySize = size;
            auto readFunc = [this, dataOffset, entrySize]() -> std::vector<uint8_t> {
              return this->readAt(dataOffset, entrySize);
            };

            entries.emplace_back(name, size, readFunc);
            found = true;
            break;
          }
        }

        if (found) break;
        // Move position by the amount read (minus overlap if necessary, but here we scan linearly)
        pos += static_cast<std::streamoff>(toRead);
      }

      if (!found) {
        throw std::runtime_error("ZIP data descriptor not found for entry: " + name);
      }
    }
  }

  return entries;
}

}  // namespace splat
