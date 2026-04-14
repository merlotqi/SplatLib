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
#include <filesystem>
#include <string>

namespace splat {

namespace {

/** ZIP entry names: forward slashes, UTF-8 bytes (LFH/CDR use UTF-8 flag). */
std::string zipEntryUtf8(const std::filesystem::path& entry) {
  const std::filesystem::path norm = entry.lexically_normal();
  std::string out;
  bool first = true;
  for (const auto& part : norm) {
    if (!first) {
      out += '/';
    }
    first = false;
    out += part.u8string();
  }
  return out;
}

}  // namespace

namespace zip_constants {

static constexpr uint32_t SIG_LFH = 0x04034b50;
static constexpr uint32_t SIG_CDR = 0x02014b50;
static constexpr uint32_t SIG_EOCD = 0x06054b50;
static constexpr uint32_t SIG_DESCRIPTOR = 0x08074b50;
static constexpr uint16_t LFH_FIXED_SIZE = 30;
static constexpr uint16_t CDR_FIXED_SIZE = 46;

}  // namespace zip_constants

ZipWriter::ZipWriter(const std::filesystem::path& filename) {
  file_.open(filename, std::ios::binary | std::ios::out);
  if (!file_) {
    throw std::runtime_error("Failed to open zip file: " + filename.u8string());
  }

  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm* tm = std::localtime(&t);

  dosTime_ = static_cast<uint16_t>((tm->tm_hour << 11) | (tm->tm_min << 5) | (tm->tm_sec / 2));

  dosDate_ = static_cast<uint16_t>(((tm->tm_year + 1900 - 1980) << 9) | ((tm->tm_mon + 1) << 5) | tm->tm_mday);
}

ZipWriter::~ZipWriter() { close(); }

ZipWriter::ZipWriter(ZipWriter&& other) noexcept { *this = std::move(other); }

ZipWriter& ZipWriter::operator=(ZipWriter&& other) noexcept {
  if (this != &other) {
    close();
    file_ = std::move(other.file_);
    files_ = std::move(other.files_);
    file_open_ = other.file_open_;
    dosTime_ = other.dosTime_;
    dosDate_ = other.dosDate_;
    other.file_open_ = false;
  }
  return *this;
}

void ZipWriter::start(const std::filesystem::path& filename) {
  if (file_open_) {
    finishCurrentFile();
  }

  const std::string utf8 = zipEntryUtf8(filename);
  std::vector<uint8_t> name(utf8.begin(), utf8.end());
  writeLocalFileHeader(name);
  file_open_ = true;
}

void ZipWriter::write(const uint8_t* data, size_t length) {
  if (!file_open_) {
    throw std::runtime_error("write() called before start()");
  }

  FileInfo& f = files_.back();
  f.crc.update(data, length);
  f.sizeBytes += static_cast<uint32_t>(length);
  file_.write(reinterpret_cast<const char*>(data), length);
}

void ZipWriter::write(const std::vector<uint8_t>& data) { write(data.data(), data.size()); }

void ZipWriter::close() {
  if (!file_.is_open()) {
    return;
  }

  finishCurrentFile();

  uint32_t central_dir_offset = static_cast<uint32_t>(file_.tellp());

  /* ---- Central Directory ---- */
  for (const auto& f : files_) {
    writeUint32LE(zip_constants::SIG_CDR);
    writeUint16LE(20);     // version made by
    writeUint16LE(20);     // version needed
    writeUint16LE(0x800);  // UTF-8 ONLY (no bit 3!)
    writeUint16LE(0);      // STORED
    writeUint16LE(dosTime_);
    writeUint16LE(dosDate_);
    writeUint32LE(f.crc.value());
    writeUint32LE(f.sizeBytes);
    writeUint32LE(f.sizeBytes);
    writeUint16LE(static_cast<uint16_t>(f.filenameBuf.size()));
    writeUint16LE(0);
    writeUint16LE(0);
    writeUint16LE(0);
    writeUint16LE(0);
    writeUint32LE(0);
    writeUint32LE(f.localHeaderOffset);

    file_.write(reinterpret_cast<const char*>(f.filenameBuf.data()), f.filenameBuf.size());
  }

  uint32_t central_dir_size = static_cast<uint32_t>(file_.tellp()) - central_dir_offset;

  /* ---- EOCD ---- */
  writeUint32LE(zip_constants::SIG_EOCD);
  writeUint16LE(0);
  writeUint16LE(0);
  writeUint16LE(static_cast<uint16_t>(files_.size()));
  writeUint16LE(static_cast<uint16_t>(files_.size()));
  writeUint32LE(central_dir_size);
  writeUint32LE(central_dir_offset);
  writeUint16LE(0);

  file_.close();
}

void ZipWriter::writeFile(const std::filesystem::path& filename, const std::string& content) {
  start(filename);
  write(reinterpret_cast<const uint8_t*>(content.data()), content.size());
}

void ZipWriter::writeFile(const std::filesystem::path& filename, const std::vector<uint8_t>& content) {
  start(filename);
  write(content);
}

void ZipWriter::writeFile(const std::filesystem::path& filename, const std::vector<std::vector<uint8_t>>& content) {
  start(filename);
  for (const auto& chunk : content) {
    write(chunk);
  }
}

void ZipWriter::writeUint16LE(uint16_t value) { file_.write(reinterpret_cast<const char*>(&value), 2); }

void ZipWriter::writeUint32LE(uint32_t value) {
  uint8_t buf[4] = {static_cast<uint8_t>(value & 0xFF), static_cast<uint8_t>((value >> 8) & 0xFF),
                    static_cast<uint8_t>((value >> 16) & 0xFF), static_cast<uint8_t>((value >> 24) & 0xFF)};
  file_.write(reinterpret_cast<const char*>(buf), 4);
}

void ZipWriter::writeLocalFileHeader(const std::vector<uint8_t>& filenameBuf) {
  uint32_t offset = static_cast<uint32_t>(file_.tellp());

  writeUint32LE(zip_constants::SIG_LFH);
  writeUint16LE(20);
  writeUint16LE(0x8 | 0x800);  // Data Descriptor + UTF-8
  writeUint16LE(0);            // STORED
  writeUint16LE(dosTime_);
  writeUint16LE(dosDate_);
  writeUint32LE(0);
  writeUint32LE(0);
  writeUint32LE(0);
  writeUint16LE(static_cast<uint16_t>(filenameBuf.size()));
  writeUint16LE(0);

  file_.write(reinterpret_cast<const char*>(filenameBuf.data()), filenameBuf.size());

  FileInfo info;
  info.localHeaderOffset = offset;
  info.filenameBuf = filenameBuf;
  files_.push_back(info);
}

void ZipWriter::writeDataDescriptor() {
  const FileInfo& f = files_.back();
  writeUint32LE(zip_constants::SIG_DESCRIPTOR);
  writeUint32LE(f.crc.value());
  writeUint32LE(f.sizeBytes);
  writeUint32LE(f.sizeBytes);
}

void ZipWriter::finishCurrentFile() {
  if (!file_open_) {
    return;
  }
  writeDataDescriptor();
  file_open_ = false;
}

}  // namespace splat
