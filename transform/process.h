/***********************************************************************************
 *
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
 *
 ***********************************************************************************/

#include <splat/models/data-table.h>

#include <Eigen/Dense>
#include <string>
#include <variant>

namespace splat {

struct Translate {
  Eigen::Vector3f value;
};

struct Rotate {
  Eigen::Vector3f value;
};

struct Scale {
  float value;
};

struct FilterNaN {};

struct FilterByValue {
  std::string columnName;
  std::string comparator;  // lt, lte, gt, gte, eq, neq
  float value;
};

struct FilterBands {
  int value;  // 0, 1, 2, 3
};

struct FilterBox {
  Eigen::Vector3f min;
  Eigen::Vector3f max;
};

struct FilterSphere {
  Eigen::Vector3f center;
  float radius;
};

struct Param {
  std::string name;
  std::string value;
};

struct Lod {
  int value;
};

using ProcessAction =
    std::variant<Translate, Rotate, Scale, FilterNaN, FilterByValue, FilterBands, FilterBox, FilterSphere, Param, Lod>;

std::unique_ptr<DataTable> processDataTable(DataTable* dataTable, const std::vector<ProcessAction>& processActions);

}  // namespace splat
