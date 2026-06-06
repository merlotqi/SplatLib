#include <splat/gpu/gpu_compute.h>
#include <splat/spatial/kmeans.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <vector>

namespace splat {

namespace {

static std::vector<std::vector<int>> groupLabels(const std::vector<uint32_t>& labels, int k) {
  std::vector<std::vector<int>> groups(k);
  for (uint32_t i = 0; i < labels.size(); ++i) {
    groups[labels[i]].push_back(static_cast<int>(i));
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
    if (value < m) {
      m = value;
    }
    if (value > M) {
      M = value;
    }
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
    dataTable->getRow(static_cast<size_t>(cluster[i]), dataRow);
    for (size_t j = 0; j < keys.size(); ++j) {
      const auto& key = keys[j];
      row[key] += dataRow[key];
    }
  }

  if (!cluster.empty()) {
    for (size_t i = 0; i < keys.size(); ++i) {
      row[keys[i]] /= static_cast<float>(cluster.size());
    }
  }
}

}  // namespace

std::pair<std::unique_ptr<SplatCloud>, std::vector<uint32_t>> kmeans(SplatCloud* points, size_t k, size_t iterations) {
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

  const uint32_t N = static_cast<uint32_t>(points->getNumRows());
  const uint32_t K = static_cast<uint32_t>(k);
  const uint32_t D = static_cast<uint32_t>(points->getNumColumns());

  std::vector<float> h_points(static_cast<size_t>(N) * D);
  std::vector<float> h_centroids(static_cast<size_t>(K) * D);
  std::vector<float> h_centroid_norms(K);
  std::vector<uint32_t> h_results(N);

  std::random_device rd;
  std::mt19937 gen(rd());

  const auto start_total = std::chrono::high_resolution_clock::now();

  while (!converged) {
    const auto start_iter = std::chrono::high_resolution_clock::now();

    for (uint32_t d = 0; d < D; ++d) {
      const auto& colData = points->getColumn(d).asVector<float>();
      std::memcpy(&h_points[static_cast<size_t>(d) * N], colData.data(), static_cast<size_t>(N) * sizeof(float));
    }

    for (uint32_t d = 0; d < D; ++d) {
      const auto& colData = centroids->getColumn(d).asVector<float>();
      for (uint32_t c = 0; c < K; ++c) {
        h_centroids[static_cast<size_t>(c) + static_cast<size_t>(d) * K] = colData[c];
      }
    }

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
    std::memcpy(labels.data(), h_results.data(), static_cast<size_t>(N) * sizeof(uint32_t));

    const auto mid_iter = std::chrono::high_resolution_clock::now();

    auto groups = groupLabels(labels, static_cast<int>(k));
    bool centroidChanged = false;

    for (size_t i = 0; i < centroids->getNumRows(); ++i) {
      if (groups[i].empty()) {
        std::uniform_int_distribution<size_t> dis(0, points->getNumRows() - 1);
        const auto idx = dis(gen);
        points->getRow(idx, row);
        centroids->setRow(i, row);
        centroidChanged = true;
      } else {
        std::map<std::string, float> new_row;
        calcAverage(points, groups[i], new_row);
        for (const auto& item : new_row) {
          if (std::abs(item.second - centroids->getColumnByName(item.first).getValue<float>(i)) > 0.001f) {
            centroidChanged = true;
          }
          centroids->getColumnByName(item.first).setValue<float>(i, item.second);
        }
      }
    }

    const auto end_iter = std::chrono::high_resolution_clock::now();
    const auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(mid_iter - start_iter);
    const auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_iter - mid_iter);

    std::cout << "  Iteration " << (steps + 1) << ": GPU=" << gpu_time.count() << "ms, CPU=" << cpu_time.count() << "ms"
              << "\n";

    if (!centroidChanged || steps >= iterations) {
      converged = true;
    }

    ++steps;
  }

  const auto end_total = std::chrono::high_resolution_clock::now();
  const auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total);

  std::cout << "k-means completed in " << total_time.count() << "ms, " << steps << " iterations\n";
  std::cout << "GPU Backend: " << gpu::getBackendName() << "\n";

  return {std::move(centroids), labels};
}

}  // namespace splat
