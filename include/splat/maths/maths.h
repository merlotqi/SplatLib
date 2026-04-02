/**
 * @file splat/maths/maths.h
 * @brief Math utilities (sigmoid, PRNG) for splat attributes.
 *
 */
 
#pragma once

#include <absl/numeric/bits.h>

#include <cassert>
#include <cmath>
#include <cstdint>

namespace splat {

/**
 * @brief Sigmoid activation function
 *
 * Computes the sigmoid (logistic) function: σ(x) = 1 / (1 + exp(-x))
 * This function maps real values to the range (0, 1) and is commonly used
 * in machine learning for probabilities and activations.
 *
 * @tparam T Numeric type (float, double, etc.)
 * @param x Input value
 * @return Sigmoid-transformed value in range (0, 1)
 *
 * @note For Gaussian splatting, this is typically used to convert raw opacity
 *       values to valid opacity in [0, 1] range.
 * @note Uses double-precision exponential for accuracy even with float input.
 */
template <typename T>
inline T sigmoid(T x) {
  return 1.0 / (1.0 + std::exp(-x));
}

/**
 * @brief Simple deterministic pseudo-random number generator
 *
 * Generates a pseudo-random float in the range [0, 1) using a linear congruential
 * generator (LCG). The generator is deterministic and thread-unsafe due to static
 * internal state.
 *
 * @return Random float in range [0, 1)
 *
 * @note This is a simple PRNG suitable for debugging, testing, or situations
 *       where reproducibility is more important than statistical quality.
 * @note Not suitable for cryptographic or high-quality statistical applications.
 * @note The generator uses the constants from POSIX rand() implementation:
 *       multiplier = 1103515245, increment = 12345, modulus = 2^31.
 * @note Thread-unsafe: uses static internal state.
 */
inline float simple_random() {
  static unsigned int seed = 42;  ///< Initial seed (can be changed for different sequences)
  seed = (1103515245ULL * seed + 12345) & 0x7FFFFFFF;
  return (float)seed / (float)0x7FFFFFFF;
}

/** All 64 bits set (as unsigned 32-bit) */
static constexpr uint32_t SOLID_MASK = 0xFFFFFFFFu;

/**
 * Solid leaf node marker: childMask = 0xFF, baseOffset = 0.
 * This is unambiguous because BFS layout guarantees children always come after
 * their parent, so baseOffset = 0 is never valid for an interior node.
 */
static constexpr uint32_t SOLID_LEAF_MARKER = 0xFF000000u;

/**
 * Check if a voxel mask represents a solid block (all 64 bits set).
 *
 * @param lo - Lower 32 bits of mask
 * @param hi - Upper 32 bits of mask
 * @returns True if all 64 voxels are solid
 */
inline bool isSolid(uint32_t lo, uint32_t hi) { return lo == SOLID_MASK && hi == SOLID_MASK; }

/**
 * Check if a voxel mask represents an empty block (no bits set).
 *
 * @param lo - Lower 32 bits of mask
 * @param hi - Upper 32 bits of mask
 * @returns True if all 64 voxels are empty
 */
inline bool isEmpty(uint32_t lo, uint32_t hi) { return lo == 0 && hi == 0; }

/**
 * Get the offset to a child node given a parent's child mask and octant.
 * Uses absl::popcount to count how many children come before this octant.
 *
 * @param mask - 8-bit child mask from parent node
 * @param octant - Octant index (0-7)
 * @returns Offset from base child pointer
 */
inline size_t getChildOffset(uint8_t mask, int octant) {
  assert(octant >= 0 && octant < 8 && "Octant must be between 0 and 7");
  uint8_t prefix = static_cast<uint8_t>((1U << octant) - 1);
  uint8_t masked = static_cast<uint8_t>(mask & prefix);
  return absl::popcount(masked);
}

/**
 * @brief Compute the maximum of multiple values
 *
 * Uses fold expression to find the maximum value among all arguments.
 * Works with any comparable type (int, float, double, etc.).
 *
 * @tparam T Type of the values
 * @tparam Ts Types of additional values
 * @param first First value to compare
 * @param args Additional values to compare
 * @return Maximum value among all arguments
 *
 * @note This function is useful for finding the maximum dimension or bound
 *       when working with multiple values in Gaussian splatting operations.
 */
template <typename T, typename... Ts>
T maxs(T first, Ts... args) {
  T result = first;
  ((result = (args > result ? args : result)), ...);
  return result;
}

/**
 * @brief Compute the minimum of multiple values
 *
 * Uses fold expression to find the minimum value among all arguments.
 * Works with any comparable type (int, float, double, etc.).
 *
 * @tparam T Type of the values
 * @tparam Ts Types of additional values
 * @param first First value to compare
 * @param args Additional values to compare
 * @return Minimum value among all arguments
 *
 * @note This function is useful for finding the minimum dimension or bound
 *       when working with multiple values in Gaussian splatting operations.
 */
template <typename T, typename... Ts>
T mins(T first, Ts... args) {
  T result = first;
  ((result = (args < result ? args : result)), ...);
  return result;
}

}  // namespace splat
