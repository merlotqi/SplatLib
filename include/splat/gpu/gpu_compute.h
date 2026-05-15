/**
 * GPU Compute Abstraction Layer
 * 
 * This file provides platform-agnostic GPU compute interfaces
 * for CUDA and Metal backends.
 */

#pragma once

#include <cstdint>

namespace splat {
namespace gpu {

/**
 * @brief Compute centroid norms for k-means clustering
 * 
 * @param centroids Input centroids in column-major format [K×D]
 * @param K Number of centroids
 * @param D Number of dimensions
 * @param norms Output norms [K]
 * @return true on success, false on failure
 */
bool computeCentroidNorms(const float* centroids, uint32_t K, uint32_t D, float* norms);

/**
 * @brief Assign points to nearest centroids for k-means clustering
 * 
 * @param points Input points in column-major format [N×D]
 * @param centroids Centroid positions in column-major format [K×D]
 * @param centroid_norms Pre-computed norms of centroids [K]
 * @param N Number of points
 * @param K Number of centroids
 * @param D Number of dimensions
 * @param results Output cluster assignments [N]
 * @return true on success, false on failure
 */
bool assignPointsToCentroids(const float* points, const float* centroids, 
                            const float* centroid_norms,
                            uint32_t N, uint32_t K, uint32_t D,
                            uint32_t* results);

/**
 * @brief Get backend type
 * @return String describing the GPU backend (e.g., "CUDA", "Metal")
 */
const char* getBackendName();

}  // namespace gpu
}  // namespace splat
