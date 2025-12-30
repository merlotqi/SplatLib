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

#pragma once

#include <functional>
#include <memory>

namespace splat {

struct KdTreeNode {
  size_t index;
  size_t count;
  std::unique_ptr<KdTreeNode> left;
  std::unique_ptr<KdTreeNode> right;

  KdTreeNode(size_t index, size_t count, std::unique_ptr<KdTreeNode> left, std::unique_ptr<KdTreeNode> right)
      : index(index), count(count), left(std::move(left)), right(std::move(right)) {}
};

class DataTable;

class KdTree {
  DataTable* centroids;
  std::unique_ptr<KdTreeNode> root;

  std::unique_ptr<KdTreeNode> build(std::vector<size_t>& indices, size_t start, size_t end, size_t depth);

 public:
  KdTree(DataTable* table);

  enum {
    index,
    distanceSqr,
    count,
    findNearestMaxIndex = 3
  };
  std::tuple<int, float, size_t> findNearest(const std::vector<float>& point,
                                             std::function<bool(size_t)> filterFunc = nullptr);
};

}  // namespace splat
