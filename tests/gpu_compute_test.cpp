/**
 * GPU Compute Backend Test Example
 * 
 * This demonstrates how to use the GPU compute abstraction layer
 * with different backends (CUDA, Metal, or CPU).
 */

#include <splat/gpu/gpu_compute.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

int main() {
    std::cout << "=== GPU Compute Backend Test ===" << std::endl;
    std::cout << "Using backend: " << splat::gpu::getBackendName() << std::endl;
    std::cout << std::endl;

    // Test parameters
    const uint32_t K = 10;  // Number of centroids
    const uint32_t D = 3;   // Dimensions (x, y, z)
    const uint32_t N = 1000; // Number of points

    // Allocate host memory
    std::vector<float> centroids(K * D);
    std::vector<float> points(N * D);
    std::vector<float> centroid_norms(K);
    std::vector<uint32_t> assignments(N);

    // Initialize test data (column-major layout)
    // Centroids: random positions
    for (uint32_t i = 0; i < K * D; i++) {
        centroids[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }

    // Points: random positions around centroids
    for (uint32_t i = 0; i < N * D; i++) {
        points[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
    }

    // Test 1: Compute centroid norms
    std::cout << "Test 1: Computing centroid norms..." << std::endl;
    if (!splat::gpu::computeCentroidNorms(centroids.data(), K, D, centroid_norms.data())) {
        std::cerr << "ERROR: Failed to compute centroid norms" << std::endl;
        return 1;
    }

    // Verify norms
    bool norms_ok = true;
    for (uint32_t k = 0; k < K; k++) {
        float expected_norm = 0.0f;
        for (uint32_t d = 0; d < D; d++) {
            float val = centroids[k + d * K];
            expected_norm += val * val;
        }
        float tolerance = 1e-5f * expected_norm;
        if (std::fabs(centroid_norms[k] - expected_norm) > tolerance) {
            norms_ok = false;
            std::cerr << "Norm mismatch at centroid " << k << ": "
                      << centroid_norms[k] << " vs " << expected_norm << std::endl;
            break;
        }
    }

    if (norms_ok) {
        std::cout << "✓ Centroid norms are correct" << std::endl;
    } else {
        std::cerr << "✗ Centroid norms verification failed" << std::endl;
        return 1;
    }

    // Test 2: Assign points to centroids
    std::cout << "\nTest 2: Assigning points to centroids..." << std::endl;
    if (!splat::gpu::assignPointsToCentroids(points.data(), centroids.data(), 
                                              centroid_norms.data(),
                                              N, K, D, assignments.data())) {
        std::cerr << "ERROR: Failed to assign points" << std::endl;
        return 1;
    }

    // Verify assignments
    bool assignments_ok = true;
    for (uint32_t n = 0; n < N && n < 10; n++) {  // Check first 10 points
        uint32_t assigned_cluster = assignments[n];
        if (assigned_cluster >= K) {
            assignments_ok = false;
            std::cerr << "Invalid cluster assignment: " << assigned_cluster 
                      << " >= " << K << std::endl;
            break;
        }

        // Verify this is indeed the nearest centroid
        float best_dist = 3.40282e+38f;
        uint32_t best_k = 0;

        for (uint32_t k = 0; k < K; k++) {
            float dist = centroid_norms[k];
            for (uint32_t d = 0; d < D; d++) {
                float p = points[d * N + n];
                float c = centroids[k + d * K];
                dist += p * p - 2.0f * p * c;
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }

        if (best_k != assigned_cluster) {
            std::cerr << "Suboptimal assignment for point " << n << ": "
                      << assigned_cluster << " vs best " << best_k << std::endl;
            assignments_ok = false;
            break;
        }
    }

    if (assignments_ok) {
        std::cout << "✓ Point assignments are correct" << std::endl;
    } else {
        std::cerr << "✗ Point assignment verification failed" << std::endl;
        return 1;
    }

    // Statistics
    std::cout << "\nTest Statistics:" << std::endl;
    std::cout << "  Backend: " << splat::gpu::getBackendName() << std::endl;
    std::cout << "  Centroids: " << K << std::endl;
    std::cout << "  Dimensions: " << D << std::endl;
    std::cout << "  Points: " << N << std::endl;
    
    // Count assignments per cluster
    std::vector<uint32_t> cluster_counts(K, 0);
    for (uint32_t n = 0; n < N; n++) {
        cluster_counts[assignments[n]]++;
    }

    std::cout << "  Cluster distribution: ";
    for (uint32_t k = 0; k < K; k++) {
        if (k > 0) std::cout << ", ";
        std::cout << cluster_counts[k];
    }
    std::cout << std::endl;

    std::cout << "\n✓ All tests passed!" << std::endl;
    return 0;
}

/*
Build and run instructions:

For CUDA backend:
  cmake -B build_cuda -DSPLAT_USE_CUDA=ON -DSPLAT_USE_METAL=OFF
  cmake --build build_cuda
  cd build/tests && ./gpu_compute_test

For Metal backend (macOS):
  cmake -B build_metal -DSPLAT_USE_CUDA=OFF -DSPLAT_USE_METAL=ON
  cmake --build build_metal
  cd build/tests && ./gpu_compute_test

For CPU backend:
  cmake -B build_cpu -DSPLAT_USE_CUDA=OFF -DSPLAT_USE_METAL=OFF
  cmake --build build_cpu
  cd build/tests && ./gpu_compute_test
*/
