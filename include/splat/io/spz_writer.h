/**
 * @file splat/io/spz_writer.h
 * @brief Write .spz compressed Gaussian splat files (v4 NGSP format).
 */
#pragma once

#include <cstdint>
#include <filesystem>

namespace splat {

class DataTable;

struct SpzWriteOptions {
  int fractionalBits = 12;
  uint8_t sh1Bits = 5;
  uint8_t shRestBits = 4;
};

void writeSpz(const std::filesystem::path& filename,
              const DataTable& dataTable,
              const SpzWriteOptions& options = {});

}  // namespace splat
