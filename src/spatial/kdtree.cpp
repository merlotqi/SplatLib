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

#include <splat/models/data-table.h>
#include <splat/spatial/kdtree.h>

#include <numeric>


namespace splat {

KdTree::KdTree(DataTable* table) : centroids(table) {
  assert(table);
  std::vector<size_t> indices(centroids->getNumRows());
  std::iota(indices.begin(), indices.end(), 0);
  this->root = build(indices, 0, indices.size(), 0);
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

std::unique_ptr<KdTreeNode> KdTree::build(std::vector<size_t>& indices, size_t start, size_t end, size_t depth) {
  const size_t length = end - start;
  if (length == 0) {
    return nullptr;
  }

  const size_t axis = depth % centroids->getNumColumns();
  auto&& values_column = centroids->getColumn(axis);

  std::sort(indices.begin(), indices.end(), [&values_column](size_t a, size_t b) {
    return values_column.getValue<float>(a) < values_column.getValue<float>(b);
  });

  const size_t midOffset = length >> 1;
  const size_t mid = start + midOffset;

  if (indices.size() == 1) {
    return std::make_unique<KdTreeNode>(indices[0], 1, nullptr, nullptr);
  } else if (indices.size() == 2) {
    if (length == 2) {
      auto right_node = std::make_unique<KdTreeNode>(indices[mid], 1, nullptr, nullptr);
      return std::make_unique<KdTreeNode>(indices[start], 2, nullptr, std::move(right_node));
    }
  }

  auto left = build(indices, start, mid, depth + 1);
  auto right = build(indices, mid + 1, end, depth + 1);

  size_t count = 1;
  count += left ? left->count : 0;
  count += right ? right->count : 0;

  return std::make_unique<KdTreeNode>(indices[mid], count, std::move(left), std::move(right));
}

}  // namespace splat
