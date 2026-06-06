/**
 * @file splat/spatial/octree.h
 * @brief Octree structures for 3D partitioning.
 *
 * References:
 * - [Octree](https://en.wikipedia.org/wiki/Octree)
 */

#pragma once

#include <vector>

namespace splat {

class SplatCloud;

/**
 * Sorts the provided indices by visibility score (descending order).
 *
 * Visibility is computed as: linear_opacity * volume
 * where:
 * - linear_opacity = sigmoid(opacity) = 1 / (1 + exp(-opacity))
 * - volume = exp(scale_0) * exp(scale_1) * exp(scale_2)
 *
 * After calling this function, indices[0] will contain the index of the most
 * visible splat, indices[1] the second most visible, and so on.
 *
 * @param dataTable - The SplatCloud containing splat data.
 * @param indices - Array of indices to sort in-place.
 */
void sortByVisibility(const SplatCloud* dataTable, std::vector<unsigned int>& indices);

}  // namespace splat
