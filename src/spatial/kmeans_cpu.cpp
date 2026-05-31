/**
 * Fallback CPU Implementation for GPU Compute (when no GPU backend is available)
 */

#include <splat/gpu/gpu_compute.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <limits>

namespace splat {
namespace gpu {

bool computeCentroidNorms(const float* centroids, uint32_t K, uint32_t D, float* norms) {
    // CPU-based implementation
    for (uint32_t k = 0; k < K; k++) {
        float norm = 0.0f;
        for (uint32_t d = 0; d < D; d++) {
            float val = centroids[k + d * K];
            norm += val * val;
        }
        norms[k] = norm;
    }
    return true;
}

bool assignPointsToCentroids(const float* points, const float* centroids, 
                            const float* centroid_norms,
                            uint32_t N, uint32_t K, uint32_t D,
                            uint32_t* results)
{
    // CPU-based implementation
    for (uint32_t pt = 0; pt < N; pt++) {
        float minDist = std::numeric_limits<float>::infinity();
        uint32_t bestIdx = 0;
        
        // Compute point norm
        float pointNorm = 0.0f;
        for (uint32_t d = 0; d < D; d++) {
            float p = points[d * N + pt];
            pointNorm += p * p;
        }
        
        // Find nearest centroid
        for (uint32_t c = 0; c < K; c++) {
            float dist = pointNorm + centroid_norms[c];
            
            for (uint32_t d = 0; d < D; d++) {
                float p = points[d * N + pt];
                float centroid_val = centroids[c + d * K];
                dist -= 2.0f * p * centroid_val;
            }
            
            if (dist < minDist) {
                minDist = dist;
                bestIdx = c;
            }
        }
        
        results[pt] = bestIdx;
    }
    return true;
}

const char* getBackendName() {
    return "CPU (Fallback)";
}

}  // namespace gpu
}  // namespace splat
