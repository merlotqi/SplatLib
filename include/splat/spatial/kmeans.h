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

/**
 * @brief Perform k-means clustering on 3D point data.
 *
 * Implements Lloyd's algorithm for partitioning point cloud data into k clusters.
 * Each point is assigned to the cluster with the nearest centroid, with centroids
 * recomputed iteratively until convergence or maximum iterations reached.
 *
 * @param points Input DataTable containing x,y,z position columns
 * @param k Number of clusters to generate
 * @param iterations Maximum number of refinement iterations
 *
 * @return Pair containing:
 *         1. DataTable with cluster centroids (k rows, x/y/z columns)
 *         2. Vector of cluster assignments (one entry per input point, 0..k-1)
 */
std::pair<std::unique_ptr<DataTable>, std::vector<uint32_t>> kmeans(DataTable* points, size_t k, size_t iterations);

}  // namespace splat
