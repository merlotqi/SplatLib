/**
 * CUDA Backend Implementation for GPU Compute
 */

#include <splat/gpu/gpu_compute.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

namespace splat {
namespace gpu {

// CUDA kernel implementations from original kmeans.cu
__global__ void computeCentroidNormsColMajor(const float* __restrict__ centroids, 
                                             float* __restrict__ norms, 
                                             uint32_t K, uint32_t D) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    float norm = 0.0f;
    for (uint32_t d = 0; d < D; d++) {
        float val = centroids[k + d * K];
        norm += val * val;
    }
    norms[k] = norm;
}

__global__ void clusterKernelColMajor(const float* __restrict__ points, 
                                      const float* __restrict__ centroids,
                                      const float* __restrict__ centroid_norms, 
                                      uint32_t* __restrict__ results,
                                      uint32_t numPoints, uint32_t numCentroids, 
                                      uint32_t numCols) {
    uint32_t ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ptIdx >= numPoints) return;

    float minDist = 3.40282e+38f;
    uint32_t bestIdx = 0;

    float pointNorm = 0.0f;
    for (uint32_t d = 0; d < numCols; d++) {
        float p = points[d * numPoints + ptIdx];
        pointNorm += p * p;
    }

    uint32_t c = 0;
    for (; c + 7 < numCentroids; c += 8) {
        float norms[8];
        norms[0] = centroid_norms[c];
        norms[1] = centroid_norms[c + 1];
        norms[2] = centroid_norms[c + 2];
        norms[3] = centroid_norms[c + 3];
        norms[4] = centroid_norms[c + 4];
        norms[5] = centroid_norms[c + 5];
        norms[6] = centroid_norms[c + 6];
        norms[7] = centroid_norms[c + 7];

        float dist[8] = {pointNorm + norms[0], pointNorm + norms[1], pointNorm + norms[2], pointNorm + norms[3],
                         pointNorm + norms[4], pointNorm + norms[5], pointNorm + norms[6], pointNorm + norms[7]};

        for (uint32_t d = 0; d < numCols; d++) {
            float p = points[d * numPoints + ptIdx];

            float c0 = centroids[c + d * numCentroids];
            float c1 = centroids[c + 1 + d * numCentroids];
            float c2 = centroids[c + 2 + d * numCentroids];
            float c3 = centroids[c + 3 + d * numCentroids];
            float c4 = centroids[c + 4 + d * numCentroids];
            float c5 = centroids[c + 5 + d * numCentroids];
            float c6 = centroids[c + 6 + d * numCentroids];
            float c7 = centroids[c + 7 + d * numCentroids];

            dist[0] -= 2.0f * p * c0;
            dist[1] -= 2.0f * p * c1;
            dist[2] -= 2.0f * p * c2;
            dist[3] -= 2.0f * p * c3;
            dist[4] -= 2.0f * p * c4;
            dist[5] -= 2.0f * p * c5;
            dist[6] -= 2.0f * p * c6;
            dist[7] -= 2.0f * p * c7;
        }

        if (dist[0] < minDist) { minDist = dist[0]; bestIdx = c; }
        if (dist[1] < minDist) { minDist = dist[1]; bestIdx = c + 1; }
        if (dist[2] < minDist) { minDist = dist[2]; bestIdx = c + 2; }
        if (dist[3] < minDist) { minDist = dist[3]; bestIdx = c + 3; }
        if (dist[4] < minDist) { minDist = dist[4]; bestIdx = c + 4; }
        if (dist[5] < minDist) { minDist = dist[5]; bestIdx = c + 5; }
        if (dist[6] < minDist) { minDist = dist[6]; bestIdx = c + 6; }
        if (dist[7] < minDist) { minDist = dist[7]; bestIdx = c + 7; }
    }

    for (; c + 3 < numCentroids; c += 4) {
        float dist0 = pointNorm + centroid_norms[c];
        float dist1 = pointNorm + centroid_norms[c + 1];
        float dist2 = pointNorm + centroid_norms[c + 2];
        float dist3 = pointNorm + centroid_norms[c + 3];

        for (uint32_t d = 0; d < numCols; d++) {
            float p = points[d * numPoints + ptIdx];

            float c0 = centroids[c + d * numCentroids];
            float c1 = centroids[c + 1 + d * numCentroids];
            float c2 = centroids[c + 2 + d * numCentroids];
            float c3 = centroids[c + 3 + d * numCentroids];

            dist0 -= 2.0f * p * c0;
            dist1 -= 2.0f * p * c1;
            dist2 -= 2.0f * p * c2;
            dist3 -= 2.0f * p * c3;
        }

        if (dist0 < minDist) { minDist = dist0; bestIdx = c; }
        if (dist1 < minDist) { minDist = dist1; bestIdx = c + 1; }
        if (dist2 < minDist) { minDist = dist2; bestIdx = c + 2; }
        if (dist3 < minDist) { minDist = dist3; bestIdx = c + 3; }
    }

    for (; c < numCentroids; c++) {
        float dist = pointNorm + centroid_norms[c];
        for (uint32_t d = 0; d < numCols; d++) {
            float p = points[d * numPoints + ptIdx];
            float centroid_val = centroids[c + d * numCentroids];
            dist -= 2.0f * p * centroid_val;
        }

        if (dist < minDist) {
            minDist = dist;
            bestIdx = c;
        }
    }

    results[ptIdx] = bestIdx;
}

bool computeCentroidNorms(const float* centroids, uint32_t K, uint32_t D, float* norms) {
    float *d_centroids = nullptr, *d_norms = nullptr;

    if (cudaMalloc(&d_centroids, K * D * sizeof(float)) != cudaSuccess) {
        std::cerr << "CUDA: Failed to allocate centroid memory\n";
        return false;
    }
    if (cudaMalloc(&d_norms, K * sizeof(float)) != cudaSuccess) {
        std::cerr << "CUDA: Failed to allocate norms memory\n";
        cudaFree(d_centroids);
        return false;
    }

    if (cudaMemcpy(d_centroids, centroids, K * D * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "CUDA: Failed to copy centroid data\n";
        cudaFree(d_norms);
        cudaFree(d_centroids);
        return false;
    }

    dim3 blockDim(256);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x);
    computeCentroidNormsColMajor<<<gridDim, blockDim>>>(d_centroids, d_norms, K, D);

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "CUDA: Kernel execution failed\n";
        cudaFree(d_norms);
        cudaFree(d_centroids);
        return false;
    }

    if (cudaMemcpy(norms, d_norms, K * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "CUDA: Failed to copy norms result\n";
        cudaFree(d_norms);
        cudaFree(d_centroids);
        return false;
    }

    cudaFree(d_norms);
    cudaFree(d_centroids);
    return true;
}

bool assignPointsToCentroids(const float* points, const float* centroids, 
                            const float* centroid_norms,
                            uint32_t N, uint32_t K, uint32_t D,
                            uint32_t* results)
{
    float *d_points = nullptr, *d_centroids = nullptr, *d_centroid_norms = nullptr;
    uint32_t *d_results = nullptr;

    if (cudaMalloc(&d_points, N * D * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_centroids, K * D * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_centroid_norms, K * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_results, N * sizeof(uint32_t)) != cudaSuccess) {
        std::cerr << "CUDA: Failed to allocate GPU memory\n";
        cudaFree(d_points);
        cudaFree(d_centroids);
        cudaFree(d_centroid_norms);
        cudaFree(d_results);
        return false;
    }

    if (cudaMemcpy(d_points, points, N * D * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_centroids, centroids, K * D * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_centroid_norms, centroid_norms, K * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "CUDA: Failed to copy data to GPU\n";
        cudaFree(d_points);
        cudaFree(d_centroids);
        cudaFree(d_centroid_norms);
        cudaFree(d_results);
        return false;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    clusterKernelColMajor<<<blocksPerGrid, threadsPerBlock>>>(
        d_points, d_centroids, d_centroid_norms, d_results, N, K, D);

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::cerr << "CUDA: Kernel execution failed\n";
        cudaFree(d_points);
        cudaFree(d_centroids);
        cudaFree(d_centroid_norms);
        cudaFree(d_results);
        return false;
    }

    if (cudaMemcpy(results, d_results, N * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "CUDA: Failed to copy results from GPU\n";
        cudaFree(d_points);
        cudaFree(d_centroids);
        cudaFree(d_centroid_norms);
        cudaFree(d_results);
        return false;
    }

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_centroid_norms);
    cudaFree(d_results);
    return true;
}

const char* getBackendName() {
    return "CUDA";
}

}  // namespace gpu
}  // namespace splat
