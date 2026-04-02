/**
 * @file splat/spatial/kmeans.h
 * @brief k-means utilities for spatial code.
 *
 * References:
 * - [k-means](https://en.wikipedia.org/wiki/K-means_clustering)
 */
 
#pragma once

#include <splat/models/data-table.h>

namespace splat {

std::pair<std::unique_ptr<DataTable>, std::vector<uint32_t>> kmeans(DataTable* points, size_t k, size_t iterations);

}  // namespace splat
