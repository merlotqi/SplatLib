/**
 * @file splat/utils/crc.h
 * @brief CRC32 for chunk integrity.
 *
 * References:
 * - [CRC-32](https://en.wikipedia.org/wiki/Cyclic_redundancy_check)
 */
 
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace splat {

class Crc {
  uint32_t bits_{0xFFFFFFFF};
  static const std::array<uint32_t, 256> crc32_table;

 public:
  Crc() = default;
  void reset();
  void update(const uint8_t* data, std::size_t length);
  void update(const std::vector<uint8_t>& data);
  [[nodiscard]] uint32_t value() const;
};

}  // namespace splat
