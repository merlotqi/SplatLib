#include <splat/models/data-table.h>
#include <splat/spatial/kdtree.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace {

std::vector<size_t> bruteForceKNearest(const std::vector<float>& xs, const std::vector<float>& ys,
                                       const std::vector<float>& zs, const float* point, size_t k) {
  std::vector<std::pair<float, size_t>> distances;
  distances.reserve(xs.size());
  for (size_t i = 0; i < xs.size(); ++i) {
    const float dx = xs[i] - point[0];
    const float dy = ys[i] - point[1];
    const float dz = zs[i] - point[2];
    distances.emplace_back(dx * dx + dy * dy + dz * dz, i);
  }

  std::sort(distances.begin(), distances.end());

  std::vector<size_t> result;
  result.reserve(k);
  for (size_t i = 0; i < std::min(k, distances.size()); ++i) {
    result.push_back(distances[i].second);
  }
  return result;
}

}  // namespace

int main() {
  std::vector<float> xs = {0.0f, 3.0f, 1.0f, -2.0f, 0.5f, 2.0f};
  std::vector<float> ys = {0.0f, 1.0f, 2.0f, -1.0f, 0.25f, 2.0f};
  std::vector<float> zs = {0.0f, 1.0f, 0.5f, -1.0f, 0.25f, 3.0f};

  splat::DataTable table({
      {"x", xs},
      {"y", ys},
      {"z", zs},
  });
  splat::KdTree kd(&table);

  const float point[3] = {0.25f, 0.2f, 0.1f};
  std::vector<size_t> actual;
  kd.findKNearest(point, 3, actual);

  const std::vector<size_t> expected = bruteForceKNearest(xs, ys, zs, point, 3);
  assert(actual == expected);

  actual.clear();
  kd.findKNearest(point, 32, actual);
  assert(actual.size() == xs.size());

  return 0;
}
