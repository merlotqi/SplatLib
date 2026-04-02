/**
 * @file splat/op/morton-order.h
 * @brief Morton (Z-order) keys for spatial locality.
 *
 * References:
 * - [Z-order / Morton curve](https://en.wikipedia.org/wiki/Z-order_curve)
 */
 
#pragma once

#include <absl/types/span.h>

#include <array>
#include <cstdint>

namespace splat {

class DataTable;

/**
 * @brief Sort Gaussian splats in Morton order (Z-order curve) for memory locality
 *
 * This function computes a Morton ordering (Z-order curve) of the Gaussian splats
 * based on their 3D positions to improve cache locality and memory access patterns.
 * The Morton code interleaves the bits of the 3D coordinates to create a 1D ordering
 * that preserves spatial proximity in multi-dimensional space.
 *
 * @param dataTable Pointer to the DataTable containing Gaussian splat data.
 *                  Expected to have at least 'x', 'y', and 'z' columns representing
 *                  the 3D positions of the splats.
 * @param indices Output span that will be filled with the sorted indices in Morton order.
 *                Must be pre-allocated with size equal to the number of rows in dataTable.
 *                On output, indices[i] contains the original index of the i-th splat
 *                in Morton order.
 *
 * @note The function assumes the position coordinates are normalized or bounded.
 * @note Morton ordering is particularly effective for octree-based rendering and
 *       improves performance for spatially coherent operations.
 * @note This is a key optimization for real-time Gaussian splatting rendering pipelines.
 */
void sortMortonOrder(const DataTable* dataTable, absl::Span<uint32_t> indices);

/**
 * Encode block coordinates to Morton code (17 bits per axis = 51 bits total).
 * Supports up to 131,072 blocks per axis.
 *
 * @param x - Block X coordinate
 * @param y - Block Y coordinate
 * @param z - Block Z coordinate
 * @returns Morton code with interleaved bits: ...z2y2x2 z1y1x1 z0y0x0
 */
inline uint32_t xyzToMorton(int x, int y, int z) {
  uint32_t result = 0;

  for (int i = 0; i < 17; i++) {
    uint32_t bits = 0;
    bits |= (x >> i) & 1;
    bits |= ((y >> i) & 1) << 1;
    bits |= ((z >> i) & 1) << 2;
    result |= bits << (i * 3);
  }

  return result;
}

/**
 * Decode Morton code to block coordinates.
 *
 * @param m - Morton code
 * @returns array of [x, y, z] block coordinates
 */
inline std::array<int, 3> mortonToXYZ(unsigned int m) {
  int x = 0, y = 0, z = 0;
  int bit = 1;

  while (m > 0) {
    unsigned int triplet = m % 8;

    if (triplet & 1) x |= bit;
    if (triplet & 2) y |= bit;
    if (triplet & 4) z |= bit;

    bit <<= 1;
    m = m / 8;
  }

  return {x, y, z};
}

}  // namespace splat
