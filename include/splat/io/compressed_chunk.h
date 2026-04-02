/**
 * @file splat/io/compressed_chunk.h
 * @brief Packed compressed chunk layout for splat blocks
 *
 * Defines data structures for quantized and bit-packed Gaussian splat
 * attributes used in compressed PLY format.
 */
 
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace splat {

/**
 * @brief Spatial chunk of splats with bit-packed compression
 *
 * Stores Gaussian splat attributes (position, rotation, color, scale)
 * in both unpacked (float) and packed (quantized uint32) representations.
 */
class CompressedChunk {
  std::map<std::string, std::vector<float>> data;  ///< Unpacked splat attributes
  size_t size;                                      ///< Maximum splats per chunk

 public:
  /**
   * @brief Construct chunk with given capacity
   * @param size Maximum number of splats (default 256)
   */
  CompressedChunk(size_t size = 256);

  /**
   * @brief Store decoded attributes for one splat
   * @param index Splat index within the chunk
   * @param dataMap Attribute key-value pairs (position, color, etc.)
   */
  void set(size_t index, const std::map<std::string, float>& dataMap);

  /**
   * @brief Quantize and pack attributes into compressed streams
   *
   * Converts float attributes to quantized integer codes for
   * position, rotation, color, and scale.
   */
  void pack();

  ///< Packed position codes per splat (quantized)
  std::vector<uint32_t> position;
  ///< Packed rotation codes per splat (quantized quaternion)
  std::vector<uint32_t> rotation;
  ///< Packed color/SH codes per splat (quantized)
  std::vector<uint32_t> color;
  ///< Packed scale codes per splat (quantized)
  std::vector<uint32_t> scale;
  ///< Raw float payload after pack (layout-specific)
  std::vector<float> chunkData;
};

}  // namespace splat
