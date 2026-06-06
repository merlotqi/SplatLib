#include <splat/models/splatcloud.h>
#include <splat/spatial/kdtree.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

namespace splat {

KdTree::KdTree(SplatCloud* table) : centroids(table) {
  assert(table);
  columnData.reserve(centroids->getNumColumns());
  for (size_t i = 0; i < centroids->getNumColumns(); ++i) {
    columnData.push_back(&centroids->getColumn(i).asVector<float>());
  }

  std::vector<size_t> indices(centroids->getNumRows());
  std::iota(indices.begin(), indices.end(), 0);
  this->root = build(absl::MakeSpan(indices), 0);
}

std::tuple<int, float, size_t> KdTree::findNearest(const std::vector<float>& point,
                                                   std::function<bool(size_t)> filterFunc) {
  if (!root || centroids->getNumColumns() == 0) {
    return {-1, std::numeric_limits<float>::infinity(), 0};
  }

  float mind = std::numeric_limits<float>::infinity();
  int mini = -1;
  size_t cnt = 0;

  const size_t numColumns = columnData.size();

  auto calcDistance = [&](size_t index) -> float {
    float l = 0.0f;
    for (size_t i = 0; i < numColumns; ++i) {
      float v = (*columnData[i])[index] - point[i];
      l += v * v;
    }
    return l;
  };

  auto recurse = [&](auto&& self, KdTreeNode* node, size_t axis) -> void {
    if (!node) return;

    float node_split_value = (*columnData[axis])[node->index];
    const float distance_on_axis = point[axis] - node_split_value;

    auto next = (distance_on_axis > 0) ? node->right.get() : node->left.get();
    auto other = (next == node->right.get()) ? node->left.get() : node->right.get();
    const size_t next_axis = axis + 1 < numColumns ? axis + 1 : 0;

    cnt++;

    if (next) {
      self(self, next, next_axis);
    }

    if (!filterFunc || filterFunc(node->index)) {
      const float thisd = calcDistance(node->index);
      if (thisd < mind) {
        mind = thisd;
        mini = node->index;
      }
    }

    if (distance_on_axis * distance_on_axis < mind) {
      if (other) {
        self(self, other, next_axis);
      }
    }
  };

  recurse(recurse, root.get(), 0);

  return {mini, mind, cnt};
}

std::vector<size_t> KdTree::findKNearest(const std::vector<float>& point, size_t k,
                                         std::function<bool(size_t)> filterFunc) {
  if (!root || k == 0 || centroids->getNumColumns() == 0) {
    return {};
  }
  if (point.size() < columnData.size()) {
    return {};
  }

  std::vector<size_t> out;
  findKNearest(point.data(), k, out, std::move(filterFunc));
  return out;
}

void KdTree::findKNearest(const float* point, size_t k, std::vector<size_t>& out,
                          std::function<bool(size_t)> filterFunc) {
  out.clear();
  if (!root || k == 0 || columnData.empty() || !point) {
    return;
  }

  k = std::min(k, centroids->getNumRows());
  heapDistances.assign(k, std::numeric_limits<float>::infinity());
  heapIndices.assign(k, std::numeric_limits<size_t>::max());
  size_t heapSize = 0;
  const size_t numColumns = columnData.size();

  auto calcDistance = [&](size_t index) -> float {
    float l = 0.0f;
    for (size_t i = 0; i < numColumns; ++i) {
      const float v = (*columnData[i])[index] - point[i];
      l += v * v;
    }
    return l;
  };

  auto heap_push = [&](float dist, size_t idx) {
    if (heapSize < k) {
      size_t pos = heapSize++;
      heapDistances[pos] = dist;
      heapIndices[pos] = idx;

      while (pos > 0) {
        const size_t parent = (pos - 1) >> 1;
        if (heapDistances[parent] >= heapDistances[pos]) {
          break;
        }
        std::swap(heapDistances[parent], heapDistances[pos]);
        std::swap(heapIndices[parent], heapIndices[pos]);
        pos = parent;
      }
      return;
    }

    if (dist >= heapDistances[0]) {
      return;
    }

    heapDistances[0] = dist;
    heapIndices[0] = idx;

    size_t pos = 0;
    for (;;) {
      const size_t left = pos * 2 + 1;
      const size_t right = left + 1;
      size_t largest = pos;

      if (left < heapSize && heapDistances[left] > heapDistances[largest]) largest = left;
      if (right < heapSize && heapDistances[right] > heapDistances[largest]) largest = right;
      if (largest == pos) break;

      std::swap(heapDistances[pos], heapDistances[largest]);
      std::swap(heapIndices[pos], heapIndices[largest]);
      pos = largest;
    }
  };

  auto recurse = [&](auto&& self, KdTreeNode* node, size_t axis) -> void {
    if (!node) {
      return;
    }

    const float node_split_value = (*columnData[axis])[node->index];
    const float distance_on_axis = point[axis] - node_split_value;

    KdTreeNode* next = (distance_on_axis > 0) ? node->right.get() : node->left.get();
    KdTreeNode* other = (next == node->right.get()) ? node->left.get() : node->right.get();
    const size_t next_axis = axis + 1 < numColumns ? axis + 1 : 0;

    if (next) {
      self(self, next, next_axis);
    }

    if (!filterFunc || filterFunc(node->index)) {
      heap_push(calcDistance(node->index), node->index);
    }

    const float w = heapSize < k ? std::numeric_limits<float>::infinity() : heapDistances[0];
    if (distance_on_axis * distance_on_axis < w) {
      if (other) {
        self(self, other, next_axis);
      }
    }
  };

  recurse(recurse, root.get(), 0);

  for (size_t i = 1; i < heapSize; ++i) {
    const float dist = heapDistances[i];
    const size_t idx = heapIndices[i];
    size_t j = i;
    while (j > 0 && heapDistances[j - 1] > dist) {
      heapDistances[j] = heapDistances[j - 1];
      heapIndices[j] = heapIndices[j - 1];
      --j;
    }
    heapDistances[j] = dist;
    heapIndices[j] = idx;
  }

  out.resize(heapSize);
  for (size_t i = 0; i < heapSize; ++i) {
    out[i] = heapIndices[i];
  }
}

std::unique_ptr<KdTree::KdTreeNode> KdTree::build(absl::Span<size_t> indices, size_t depth) {
  if (indices.empty()) {
    return nullptr;
  }

  const size_t axis = depth % columnData.size();
  const std::vector<float>& values_column = *columnData[axis];

  size_t mid = indices.size() >> 1;

  std::nth_element(indices.begin(), indices.begin() + mid, indices.end(),
                   [&](size_t a, size_t b) { return values_column[a] < values_column[b]; });

  size_t node_index = indices[mid];

  if (indices.size() == 1) {
    return std::make_unique<KdTreeNode>(node_index, 1, nullptr, nullptr);
  }

  if (indices.size() == 2) {
    auto left_leaf = std::make_unique<KdTreeNode>(indices[0], 1, nullptr, nullptr);
    return std::make_unique<KdTreeNode>(node_index, 2, std::move(left_leaf), nullptr);
  }

  auto left = build(indices.subspan(0, mid), depth + 1);
  auto right = build(indices.subspan(mid + 1), depth + 1);

  size_t total_count = 1;
  if (left) total_count += left->count;
  if (right) total_count += right->count;

  return std::make_unique<KdTreeNode>(node_index, total_count, std::move(left), std::move(right));
}

}  // namespace splat
