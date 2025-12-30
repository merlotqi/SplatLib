/**
 * splat - A C++ library for reading and writing 3D Gaussian Splatting (splat) files.
 *
 * This library provides functionality to convert, manipulate, and process
 * 3D Gaussian splatting data formats used in real-time neural rendering.
 *
 * This file is part of splat.
 *
 * splat is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * splat is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * For more information, visit the project's homepage or contact the author.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <splat/spatial/kmeans.h>

#include <iostream>
#include <numeric>
#include <random>

__global__ void assignment_kernel(const float* __restrict__ points, const float* __restrict__ centroids,
                                  uint32_t* __restrict__ labels, int numPoints, int k, int numCols) {
  extern __shared__ float sharedCentroids[];

  int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int localIdx = threadIdx.x;
  const int chunkSize = blockDim.x;

  float minDistanceSqr = 3.402823466e+38F;  // FLT_MAX
  uint32_t bestCentroid = 0;

  float currentPoint[64];
  if (pointIdx < numPoints) {
    for (int d = 0; d < numCols; d++) {
      currentPoint[d] = points[pointIdx * numCols + d];
    }
  }

  int numChunks = (k + chunkSize - 1) / chunkSize;
  for (int i = 0; i < numChunks; i++) {
    int centroidToLoad = i * chunkSize + localIdx;
    if (centroidToLoad < k) {
      for (int d = 0; d < numCols; d++) {
        sharedCentroids[localIdx * numCols + d] = centroids[centroidToLoad * numCols + d];
      }
    }
    __syncthreads();

    if (pointIdx < numPoints) {
      int currentChunkSize = min(chunkSize, k - i * chunkSize);
      for (int c = 0; c < currentChunkSize; c++) {
        float distSqr = 0.0f;
        for (int d = 0; d < numCols; d++) {
          float diff = currentPoint[d] - sharedCentroids[c * numCols + d];
          distSqr += diff * diff;
        }
        if (distSqr < minDistanceSqr) {
          minDistanceSqr = distSqr;
          bestCentroid = i * chunkSize + c;
        }
      }
    }
    __syncthreads();
  }

  if (pointIdx < numPoints) {
    labels[pointIdx] = bestCentroid;
  }
}

__global__ void update_sums_kernel(const float* points, const uint32_t* labels, float* centroidSums, int* counts,
                                   int numPoints, int numCols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numPoints) {
    uint32_t label = labels[idx];

    atomicAdd(&counts[label], 1);

    for (int d = 0; d < numCols; d++) {
      atomicAdd(&centroidSums[label * numCols + d], points[idx * numCols + d]);
    }
  }
}

__global__ void normalize_kernel(float* centroids, const float* centroidSums, const int* counts, int k, int numCols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < k) {
    int count = counts[idx];
    if (count > 0) {
      for (int d = 0; d < numCols; d++) {
        centroids[idx * numCols + d] = centroidSums[idx * numCols + d] / (float)count;
      }
    }
  }
}

namespace splat {

std::vector<float> interleaveDataTable(const splat::DataTable* table) {
  size_t rows = table->getNumRows();
  size_t cols = table->getNumColumns();
  std::vector<float> output(rows * cols);
  for (size_t c = 0; c < cols; ++c) {
    auto span = table->getColumn(c).asSpan<float>();
    for (size_t r = 0; r < rows; ++r) {
      output[r * cols + c] = span[r];
    }
  }
  return output;
}

static void interleaveToBuffer(const DataTable* table, std::vector<float>& buffer) {
  size_t rows = table->getNumRows();
  size_t cols = table->getNumColumns();
  buffer.resize(rows * cols);
  for (size_t c = 0; c < cols; ++c) {
    auto span = table->getColumn(c).asSpan<float>();
    for (size_t r = 0; r < rows; ++r) {
      buffer[r * cols + c] = span[r];
    }
  }
}

std::pair<std::unique_ptr<DataTable>, std::vector<uint32_t>> kmeans(DataTable* points, size_t k, size_t iterations) {
  const int numPoints = static_cast<int>(points->getNumRows());
  const int numCols = static_cast<int>(points->getNumColumns());

  if (numPoints < k) {
    std::vector<uint32_t> labels(numPoints);
    std::iota(labels.begin(), labels.end(), 0);
    return {points->clone(), labels};
  }

  float *d_points, *d_centroids, *d_centroidSums;
  uint32_t* d_labels;
  int* d_counts;

  cudaMalloc(&d_points, numPoints * numCols * sizeof(float));
  cudaMalloc(&d_centroids, k * numCols * sizeof(float));
  cudaMalloc(&d_centroidSums, k * numCols * sizeof(float));
  cudaMalloc(&d_labels, numPoints * sizeof(uint32_t));
  cudaMalloc(&d_counts, k * sizeof(int));

  std::vector<float> h_points_buffer;
  interleaveToBuffer(points, h_points_buffer);
  cudaMemcpy(d_points, h_points_buffer.data(), h_points_buffer.size() * sizeof(float), cudaMemcpyHostToDevice);

  std::unique_ptr<DataTable> centroidsTable = std::make_unique<DataTable>();
  for (auto& col : points->columns) {
    centroidsTable->addColumn({col.name, std::vector<float>(k, 0.0f)});
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0, numPoints - 1);
  Row tempRow;
  for (size_t i = 0; i < k; ++i) {
    points->getRow(dis(gen), tempRow);
    centroidsTable->setRow(i, tempRow);
  }

  std::vector<float> h_centroids_buffer;
  interleaveToBuffer(centroidsTable.get(), h_centroids_buffer);
  cudaMemcpy(d_centroids, h_centroids_buffer.data(), h_centroids_buffer.size() * sizeof(float), cudaMemcpyHostToDevice);

  std::cout << "Starting GPU K-means: " << numPoints << " points, " << k << " clusters, " << numCols << " dims."
            << std::endl;

  for (size_t step = 0; step < iterations; ++step) {
    int threads = 128;
    int blocks = (numPoints + threads - 1) / threads;
    size_t sharedMemSize = threads * numCols * sizeof(float);

    assignment_kernel<<<blocks, threads, sharedMemSize>>>(d_points, d_centroids, d_labels, numPoints, (int)k, numCols);

    cudaMemset(d_centroidSums, 0, k * numCols * sizeof(float));
    cudaMemset(d_counts, 0, k * sizeof(int));

    update_sums_kernel<<<blocks, threads>>>(d_points, d_labels, d_centroidSums, d_counts, numPoints, numCols);

    normalize_kernel<<<(k + 127) / 128, 128>>>(d_centroids, d_centroidSums, d_counts, (int)k, numCols);

    std::vector<int> h_counts(k);
    cudaMemcpy(h_counts.data(), d_counts, k * sizeof(int), cudaMemcpyDeviceToHost);

    bool anyEmpty = false;
    for (size_t i = 0; i < k; ++i) {
      if (h_counts[i] == 0) {
        points->getRow(dis(gen), tempRow);
        std::vector<float> h_random_pt(numCols);
        for (int d = 0; d < numCols; ++d) h_random_pt[d] = tempRow[points->columns[d].name];

        cudaMemcpy(d_centroids + i * numCols, h_random_pt.data(), numCols * sizeof(float), cudaMemcpyHostToDevice);
        anyEmpty = true;
      }
    }

    std::cout << "#" << std::flush;
  }
  std::cout << " done." << std::endl;

  std::vector<uint32_t> labels(numPoints);
  cudaMemcpy(labels.data(), d_labels, numPoints * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  std::vector<float> h_final_centroids(k * numCols);
  cudaMemcpy(h_final_centroids.data(), d_centroids, k * numCols * sizeof(float), cudaMemcpyDeviceToHost);

  for (int c = 0; c < numCols; ++c) {
    auto& colData = centroidsTable->getColumn(c).asVector<float>();
    for (int r = 0; r < k; ++r) {
      colData[r] = h_final_centroids[r * numCols + c];
    }
  }

  cudaFree(d_points);
  cudaFree(d_centroids);
  cudaFree(d_centroidSums);
  cudaFree(d_labels);
  cudaFree(d_counts);

  return {std::move(centroidsTable), labels};
}
}  // namespace splat