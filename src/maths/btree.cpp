#include <splat/maths/btree.h>

namespace splat {

AABB::AABB(const std::vector<double>& min, const std::vector<double>& max) : min(min), max(max) {}

int AABB::largetAxis() const {
  const size_t length = min.size();
  if (length == 0) return -1;
  auto l = -std::numeric_limits<double>::infinity();
  int result = -1;
  for (size_t i = 0; i < length; i++) {
    const double e = max[i] - min[i];
    if (e > l) {
      l = e;
      result = static_cast<int>(i);
    }
  }
  return result;
}

double AABB::largestDim() const {
  const double a = largetAxis();
  return max[a] - min[a];
}

AABB& AABB::fromCentroids(const DataTable& centroids, const std::vector<int>& indices) {
  for (size_t i = 0; i < centroids.getNumColumns(); i++) {
    const auto data = centroids.getColumn(i);
    double m = std::numeric_limits<double>::infinity();
    double n = -std::numeric_limits<double>::infinity();

    for (auto index : indices) {
      const double v = data->getValue(index);
      m = std::min(v, m);
      n = std::max(v, n);
    }
    min[i] = m;
    max[i] = n;
  }
  return *this;
}

}  // namespace splat
