/**
 * @file splat/utils/zip-reader.h
 * @brief Read ZIP archives.
 *
 * References:
 * - [ZIP APPNOTE](https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT)
 */
 
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace splat {

class ZipEntry {
 public:
  std::filesystem::path name;
  uint32_t size;                                   // Uncompressed size
  std::function<std::vector<uint8_t>()> readData;  // Lazy data read function
  ZipEntry(std::filesystem::path n, uint32_t sz, std::function<std::vector<uint8_t>()> rd)
      : name(std::move(n)), size(sz), readData(std::move(rd)) {}
};

/**
 * @brief Minimal ZIP reader supporting STORED (method 0) and data descriptors.
 * It sequentially parses Local File Headers to list entries.
 */
class ZipReader {
 private:
  std::ifstream file_;
  // Use std::streamoff for offsets; it's designed for signed differences/offsets,
  // though std::streampos is acceptable for absolute positions.
  std::streamoff cursor_ = 0;
  std::streamoff file_size_ = 0;

 public:
  explicit ZipReader(const std::filesystem::path& filename);
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
