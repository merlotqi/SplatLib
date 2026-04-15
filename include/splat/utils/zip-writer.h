/**
 * @file splat/utils/zip-writer.h
 * @brief Write ZIP archives.
 *
 * References:
 * - [ZIP APPNOTE](https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT)
 */

#pragma once

#include <splat/utils/crc.h>

#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace splat {

/**
 * @brief Represents the metadata for a single file entry within the archive.
 *
 * This structure stores file metadata used to construct the Central Directory Record (CDR)
 * in a ZIP archive. It tracks essential file information during the creation process.
 */
struct FileInfo {
  std::vector<uint8_t> filenameBuf;  ///< UTF-8 encoded filename bytes
  Crc crc;                           ///< CRC-32 checksum calculator for file integrity verification
  uint32_t sizeBytes = 0;            ///< Uncompressed size in bytes (for STORE compression method)
  uint32_t localHeaderOffset = 0;    ///< Byte offset of the Local File Header (LFH) within the archive
};

/**
 * @brief Synchronous streaming ZIP archive writer
 *
 * A RAII-based ZIP archive writer that handles ZIP format encoding for uncompressed (STORED) files
 * using data descriptors. The class manages the entire ZIP file creation process including
 * Local File Headers, file data, Data Descriptors, and Central Directory Records.
 *
 * The writer supports:
 * - Creating ZIP archives with multiple files
 * - STORE compression method (no compression)
 * - Data descriptors for streaming writes
 * - Proper ZIP64 format handling for large files
 * - DOS-compatible timestamp encoding
 *
 * @note This implementation uses synchronous I/O and is not thread-safe
 * @note Files are written with the STORE method (uncompressed) for simplicity and speed
 * @note CRC-32 checksums are computed during writing for data integrity
 */
class ZipWriter {
  std::ofstream file_;           ///< The internal file stream (owned by ZipWriter)
  std::vector<FileInfo> files_;  ///< Metadata for all files in the archive
  bool file_open_ = false;       ///< Flag indicating if a file is currently being written

  // DOS Date/Time fields, calculated once upon initialization
  uint16_t dosTime_ = 0;  ///< DOS time field (HH:MM:SS encoded)
  uint16_t dosDate_ = 0;  ///< DOS date field (YYYY:MM:DD encoded)

  /**
   * @brief Write a 16-bit unsigned integer in little-endian format
   * @param value Value to write
   */
  void writeUint16LE(uint16_t value);

  /**
   * @brief Write a 32-bit unsigned integer in little-endian format
   * @param value Value to write
   */
  void writeUint32LE(uint32_t value);

  /**
   * @brief Write Local File Header for the current file
   * @param filenameBuf UTF-8 encoded filename bytes
   */
  void writeLocalFileHeader(const std::vector<uint8_t>& filenameBuf);

  /**
   * @brief Write Data Descriptor for the current file
   *
   * Data Descriptor contains CRC-32, compressed size, and uncompressed size
   * for files where these values are not known in advance (streaming writes).
   */
  void writeDataDescriptor();

  /**
   * @brief Finalize the current file being written
   *
   * Completes the current file entry by writing the Data Descriptor
   * and updating file metadata.
   */
  void finishCurrentFile();

 public:
  /**
   * @brief Construct a ZipWriter and open the output file
   * @param filename Path to the ZIP file to create
   * @throws std::runtime_error if the file cannot be opened
   */
  explicit ZipWriter(const std::filesystem::path& filename);

  /**
   * @brief Destructor - automatically closes the archive if not already closed
   *
   * If the archive is still open, writes the Central Directory and closes the file.
   */
  ~ZipWriter();

  // Disable copy semantics
  ZipWriter(const ZipWriter&) = delete;
  ZipWriter& operator=(const ZipWriter&) = delete;

  /**
   * @brief Move constructor
   * @param other Source ZipWriter to move from
   */
  ZipWriter(ZipWriter&& other) noexcept;

  /**
   * @brief Move assignment operator
   * @param other Source ZipWriter to move from
   * @return Reference to this ZipWriter
   */
  ZipWriter& operator=(ZipWriter&& other) noexcept;

  /**
   * @brief Start writing a new file to the archive
   * @param filename Name of the file to add (relative path within the archive)
   * @throws std::runtime_error if a file is already being written or archive is closed
   *
   * Begins a new file entry by writing the Local File Header and preparing
   * to receive file data.
   */
  void start(const std::filesystem::path& filename);

  /**
   * @brief Write binary data to the current file
   * @param data Pointer to the data to write
   * @param length Number of bytes to write
   * @throws std::runtime_error if no file is currently being written
   *
   * The data is written directly to the archive and the CRC-32 is updated.
   */
  void write(const uint8_t* data, size_t length);

  /**
   * @brief Write binary data from a vector to the current file
   * @param data Vector containing the data to write
   * @throws std::runtime_error if no file is currently being written
   */
  void write(const std::vector<uint8_t>& data);

  /**
   * @brief Close the current file being written
   * @throws std::runtime_error if no file is currently being written
   *
   * Finalizes the current file entry and prepares for the next file.
   */
  void close();

  /**
   * @brief Write a text file to the archive
   * @param filename Name of the file within the archive
   * @param content String content to write
   *
   * Convenience method for adding text files. The content is converted to UTF-8.
   */
  void writeFile(const std::filesystem::path& filename, const std::string& content);

  /**
   * @brief Write a binary file to the archive
   * @param filename Name of the file within the archive
   * @param content Binary data to write
   *
   * Convenience method for adding binary files.
   */
  void writeFile(const std::filesystem::path& filename, const std::vector<uint8_t>& content);

  /**
   * @brief Write a file from multiple data chunks to the archive
   * @param filename Name of the file within the archive
   * @param content Vector of data chunks to write sequentially
   *
   * Useful for writing large files that are already split into chunks.
   */
  void writeFile(const std::filesystem::path& filename, const std::vector<std::vector<uint8_t>>& content);
};

}  // namespace splat
