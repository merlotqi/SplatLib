#include <splat/models/data-table.h>
#include <splat/spatial/kdtree.h>

#include <limits>
#include <numeric>
#include <queue>
#include <utility>

namespace splat {

KdTree::KdTree(DataTable* table) : centroids(table) {
  assert(table);
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

  const size_t numColumns = centroids->getNumColumns();

  auto calcDistance = [&](size_t index) -> float {
    float l = 0.0f;
    for (size_t i = 0; i < numColumns; ++i) {
      float v = centroids->getColumn(i).getValue<float>(index) - point[i];
      l += v * v;
    }
    return l;
  };

  std::function<void(KdTreeNode*, int)> recurse = [&](KdTreeNode* node, int depth) {
    if (!node) return;

    const size_t axis = depth % numColumns;

    float node_split_value = centroids->getColumn(axis).getValue<float>(node->index);
    const float distance_on_axis = point[axis] - node_split_value;

    auto next = (distance_on_axis > 0) ? node->right.get() : node->left.get();
    auto other = (next == node->right.get()) ? node->left.get() : node->right.get();

    cnt++;

    if (next) {
      recurse(next, depth + 1);
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
        recurse(other, depth + 1);
      }
    }
  };

  recurse(root.get(), 0);

  return {mini, mind, cnt};
}

std::vector<size_t> KdTree::findKNearest(const std::vector<float>& point, size_t k,
                                         std::function<bool(size_t)> filterFunc) {
  if (!root || k == 0 || centroids->getNumColumns() == 0) {
    return {};
  }
  if (point.size() < centroids->getNumColumns()) {
    return {};
  }

  const size_t numColumns = centroids->getNumColumns();

  auto calcDistance = [&](size_t index) -> float {
    float l = 0.0f;
    for (size_t i = 0; i < numColumns; ++i) {
      float v = centroids->getColumn(i).getValue<float>(index) - point[i];
      l += v * v;
    }
    return l;
  };

  using DistIdx = std::pair<float, size_t>;
  struct MaxDistCmp {
    bool operator()(const DistIdx& a, const DistIdx& b) const { return a.first < b.first; }
  };
  std::priority_queue<DistIdx, std::vector<DistIdx>, MaxDistCmp> heap;

  auto worst_dist_sq = [&]() -> float {
    return heap.size() < k ? std::numeric_limits<float>::infinity() : heap.top().first;
  };

  auto try_push = [&](size_t idx, float d2) {
    if (filterFunc && !filterFunc(idx)) {
      return;
    }
    if (heap.size() < k) {
      heap.push({d2, idx});
    } else if (d2 < heap.top().first) {
      heap.pop();
      heap.push({d2, idx});
    }
  };

  std::function<void(KdTreeNode*, int)> recurse = [&](KdTreeNode* node, int depth) {
    if (!node) {
      return;
    }

    const size_t axis = depth % numColumns;
    float node_split_value = centroids->getColumn(axis).getValue<float>(node->index);
    const float distance_on_axis = point[axis] - node_split_value;

    KdTreeNode* next = (distance_on_axis > 0) ? node->right.get() : node->left.get();
    KdTreeNode* other = (next == node->right.get()) ? node->left.get() : node->right.get();

    if (next) {
      recurse(next, depth + 1);
    }

    if (!filterFunc || filterFunc(node->index)) {
      try_push(node->index, calcDistance(node->index));
    }

    const float w = worst_dist_sq();
    if (distance_on_axis * distance_on_axis < w) {
      if (other) {
        recurse(other, depth + 1);
      }
    }
  };

  recurse(root.get(), 0);

  std::vector<size_t> out;
  out.reserve(heap.size());
  while (!heap.empty()) {
    out.push_back(heap.top().second);
    heap.pop();
  }
  return out;
}

std::unique_ptr<KdTree::KdTreeNode> KdTree::build(absl::Span<size_t> indices, size_t depth) {
  if (indices.empty()) {
    return nullptr;
  }

  const size_t axis = depth % centroids->getNumColumns();
  auto&& values_column = centroids->getColumn(axis);

  size_t mid = indices.size() >> 1;

  std::nth_element(indices.begin(), indices.begin() + mid, indices.end(), [&](size_t a, size_t b) {
    return values_column.getValue<float>(a) < values_column.getValue<float>(b);
  });

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
