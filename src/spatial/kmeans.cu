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

#include <splat/gpu/gpu_compute.h>
#include <splat/maths/maths.h>
#include <splat/spatial/kdtree.h>
#include <splat/spatial/kmeans.h>

#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <set>

namespace splat {

static std::vector<std::vector<int>> groupLabels(const std::vector<uint32_t>& labels, int k) {
  std::vector<std::vector<int>> groups(k);
  for (uint32_t i = 0; i < labels.size(); ++i) {
    groups[labels[i]].push_back(i);
  }
  return groups;
}

static void initializeCentroids(const SplatCloud* dataTable, SplatCloud* centroids, Row& row) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0, dataTable->getNumRows() - 1);

  std::set<size_t> chosenRows;
  for (size_t i = 0; i < centroids->getNumRows(); ++i) {
    size_t candidateRow;
    do {
      candidateRow = dis(gen);
    } while (chosenRows.count(candidateRow));

    chosenRows.insert(candidateRow);

    dataTable->getRow(candidateRow, row);
    centroids->setRow(i, row);
  }
}

static void initializeCentroids1D(const SplatCloud* dataTable, SplatCloud* centroids) {
  float m = std::numeric_limits<float>::infinity();
  float M = -std::numeric_limits<float>::infinity();

  const auto& data = dataTable->getColumn(0);
  for (size_t i = 0; i < dataTable->getNumRows(); ++i) {
    float value = data.getValue<float>(i);
    if (value < m) m = value;
    if (value > M) M = value;
  }

  auto& centroidsData = centroids->getColumn(0);
  for (size_t i = 0; i < centroids->getNumRows(); ++i) {
    float value = m + (M - m) * i / (centroids->getNumRows() - 1);
    centroidsData.setValue<float>(i, value);
  }
}

static void calcAverage(const SplatCloud* dataTable, const std::vector<int>& cluster,
                        std::map<std::string, float>& row) {
  const auto keys = dataTable->getColumnNames();

  for (size_t i = 0; i < keys.size(); ++i) {
    row[keys[i]] = 0.f;
  }

  Row dataRow;
  for (size_t i = 0; i < cluster.size(); ++i) {
    dataTable->getRow(cluster[i], dataRow);

    for (size_t j = 0; j < keys.size(); ++j) {
      const auto& key = keys[j];
      row[key] += dataRow[key];
    }
  }

  if (cluster.size() > 0) {
    for (size_t i = 0; i < keys.size(); ++i) {
      row[keys[i]] /= cluster.size();
    }
  }
}

__global__ void computeCentroidNormsColMajor(const float* __restrict__ centroids, float* __restrict__ norms, uint32_t K,
                                             uint32_t D) {
  uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= K) return;

  float norm = 0.0f;
  for (uint32_t d = 0; d < D; d++) {
    float val = centroids[k + d * K];
    norm += val * val;
  }
  norms[k] = norm;
}

__global__ void clusterKernelColMajor(const float* __restrict__ points, const float* __restrict__ centroids,
                                      const float* __restrict__ centroid_norms, uint32_t* __restrict__ results,
                                      uint32_t numPoints, uint32_t numCentroids, uint32_t numCols) {
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

    if (dist[0] < minDist) {
      minDist = dist[0];
      bestIdx = c;
    }
    if (dist[1] < minDist) {
      minDist = dist[1];
      bestIdx = c + 1;
    }
    if (dist[2] < minDist) {
      minDist = dist[2];
      bestIdx = c + 2;
    }
    if (dist[3] < minDist) {
      minDist = dist[3];
      bestIdx = c + 3;
    }
    if (dist[4] < minDist) {
      minDist = dist[4];
      bestIdx = c + 4;
    }
    if (dist[5] < minDist) {
      minDist = dist[5];
      bestIdx = c + 5;
    }
    if (dist[6] < minDist) {
      minDist = dist[6];
      bestIdx = c + 6;
    }
    if (dist[7] < minDist) {
      minDist = dist[7];
      bestIdx = c + 7;
    }
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

    if (dist0 < minDist) {
      minDist = dist0;
      bestIdx = c;
    }
    if (dist1 < minDist) {
      minDist = dist1;
      bestIdx = c + 1;
    }
    if (dist2 < minDist) {
      minDist = dist2;
      bestIdx = c + 2;
    }
    if (dist3 < minDist) {
      minDist = dist3;
      bestIdx = c + 3;
    }
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

std::pair<std::unique_ptr<SplatCloud>, std::vector<uint32_t>> kmeans(SplatCloud* points, size_t k, size_t iterations) {
  // too few data points
  if (points->getNumRows() < k) {
    std::vector<uint32_t> labels(points->getNumRows(), 0);
    std::iota(labels.begin(), labels.end(), 0);
    return {points->clone(), labels};
  }

  Row row;
  std::unique_ptr<SplatCloud> centroids = std::make_unique<SplatCloud>();
  for (auto& c : points->columns) {
    centroids->addColumn({c.name, std::vector<float>(k, 0)});
  }

  if (points->getNumColumns() == 1) {
    initializeCentroids1D(points, centroids.get());
  } else {
    initializeCentroids(points, centroids.get(), row);
  }

  std::vector<uint32_t> labels(points->getNumRows(), 0);

  bool converged = false;
  size_t steps = 0;

  std::cout << "Running k-means clustering: dims=" << points->getNumColumns() << " points=" << points->getNumRows()
            << " clusters=" << k << " iterations=" << iterations << "..." << "\n";

  const uint32_t N = points->getNumRows();
  const uint32_t K = k;
  const uint32_t D = points->getNumColumns();

  // Allocate host memory for data in column-major layout
  std::vector<float> h_points(N * D);
  std::vector<float> h_centroids(K * D);
  std::vector<float> h_centroid_norms(K);
  std::vector<uint32_t> h_results(N);

  // Create random number generator for reseeding empty clusters
  std::random_device rd;
  std::mt19937 gen(rd());

  auto start_total = std::chrono::high_resolution_clock::now();

  while (!converged) {
    auto start_iter = std::chrono::high_resolution_clock::now();

    // Copy data to column-major layout
    for (uint32_t d = 0; d < D; ++d) {
      const auto& colData = points->getColumn(d).asVector<float>();
      memcpy(&h_points[d * N], colData.data(), N * sizeof(float));
    }

    for (uint32_t d = 0; d < D; ++d) {
      const auto& colData = centroids->getColumn(d).asVector<float>();
      for (uint32_t c = 0; c < K; ++c) {
        h_centroids[c + d * K] = colData[c];
      }
    }

    // Call GPU compute functions via abstraction layer
    if (!gpu::computeCentroidNorms(h_centroids.data(), K, D, h_centroid_norms.data())) {
      std::cerr << "Error: Failed to compute centroid norms on GPU\n";
      std::vector<uint32_t> fallback_labels(points->getNumRows(), 0);
      std::iota(fallback_labels.begin(), fallback_labels.end(), 0);
      return {points->clone(), fallback_labels};
    }

    if (!gpu::assignPointsToCentroids(h_points.data(), h_centroids.data(), h_centroid_norms.data(), N, K, D,
                                      h_results.data())) {
      std::cerr << "Error: Failed to assign points to centroids on GPU\n";
      std::vector<uint32_t> fallback_labels(points->getNumRows(), 0);
      std::iota(fallback_labels.begin(), fallback_labels.end(), 0);
      return {points->clone(), fallback_labels};
    }

    labels.resize(N);
    memcpy(labels.data(), h_results.data(), N * sizeof(uint32_t));
    // ======================================================

    auto mid_iter = std::chrono::high_resolution_clock::now();

    // calculate the new centroid positions
    auto groups = groupLabels(labels, k);
    bool centroidChanged = false;

    for (size_t i = 0; i < centroids->getNumRows(); ++i) {
      if (groups[i].size() == 0) {
        // re-seed this centroid to a random point to avoid zero vector
        std::uniform_int_distribution<size_t> dis(0, points->getNumRows() - 1);
        const auto idx = dis(gen);
        points->getRow(idx, row);
        centroids->setRow(i, row);
        centroidChanged = true;
      } else {
        std::map<std::string, float> new_row;
        calcAverage(points, groups[i], new_row);

        // Update centroid with new average values
        for (uint32_t d = 0; d < D; d++) {
          row[centroids->columns[d].name] = new_row[centroids->columns[d].name];
        }
        centroids->setRow(i, row);
        // Note: centroidChanged is always true in CUDA version for simplicity
        // This differs from TypeScript version which checks for actual changes
      }
    }

    steps++;

    auto end_iter = std::chrono::high_resolution_clock::now();
    auto iter_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_iter - start_iter);
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(mid_iter - start_iter);
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_iter - mid_iter);

    if (!centroidChanged || steps >= iterations) {
      converged = true;
      std::cout << "# (converged)";
    } else {
      std::cout << "#";
    }

    if (steps % 10 == 0 || converged) {
      std::cout << " [iter " << steps << ": GPU=" << gpu_duration.count() << "ms, CPU=" << cpu_duration.count()
                << "ms, total=" << iter_duration.count() << "ms]";
    }
  }

  auto end_total = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total);

  std::cout << "\nk-means completed in " << duration.count() << "ms total (Backend: " << gpu::getBackendName() << ")\n";

  return {std::move(centroids), labels};
}

}  // namespace splat
