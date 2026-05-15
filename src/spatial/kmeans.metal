/**
 * Metal Compute Kernels for k-means
 */

#include <metal_stdlib>
using namespace metal;

/**
 * Compute norms of centroids for efficient distance calculation
 * ||c||^2 = sum(c_i^2)
 */
kernel void computeCentroidNorms(const device float* centroids [[buffer(0)]],
                                 device float* norms [[buffer(1)]],
                                 constant uint& K [[buffer(2)]],
                                 constant uint& D [[buffer(3)]],
                                 uint k [[thread_position_in_grid]])
{
    if (k >= K) return;
    
    float norm = 0.0f;
    for (uint d = 0; d < D; d++) {
        float val = centroids[k + d * K];  // Column-major layout
        norm += val * val;
    }
    norms[k] = norm;
}

/**
 * Assign each point to its nearest centroid
 * Uses optimized distance calculation: ||p - c||^2 = ||p||^2 + ||c||^2 - 2*p·c
 */
kernel void assignPointsToCentroids(const device float* points [[buffer(0)]],
                                    const device float* centroids [[buffer(1)]],
                                    const device float* centroid_norms [[buffer(2)]],
                                    device uint* results [[buffer(3)]],
                                    constant uint& N [[buffer(4)]],
                                    constant uint& K [[buffer(5)]],
                                    constant uint& D [[buffer(6)]],
                                    uint pt_idx [[thread_position_in_grid]])
{
    if (pt_idx >= N) return;
    
    float minDist = INFINITY;
    uint bestIdx = 0;
    
    // Compute point norm: ||p||^2
    float pointNorm = 0.0f;
    for (uint d = 0; d < D; d++) {
        float p = points[d * N + pt_idx];  // Column-major layout
        pointNorm += p * p;
    }
    
    // Find nearest centroid
    for (uint c = 0; c < K; c++) {
        float dist = pointNorm + centroid_norms[c];
        
        // Subtract 2*p·c term
        for (uint d = 0; d < D; d++) {
            float p = points[d * N + pt_idx];
            float centroid_val = centroids[c + d * K];
            dist -= 2.0f * p * centroid_val;
        }
        
        if (dist < minDist) {
            minDist = dist;
            bestIdx = c;
        }
    }
    
    results[pt_idx] = bestIdx;
}
