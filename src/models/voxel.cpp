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

#include <splat/models/voxel.h>

#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>

namespace splat {

VoxelMetadata::VoxelMetadata(const std::string& json) {
  nlohmann::json root(json);
  version = root["version"];

  gridBounds.min = root["gridBounds"]["min"].get<std::vector<double>>();
  gridBounds.max = root["gridBounds"]["max"].get<std::vector<double>>();

  sceneBounds.min = root["sceneBounds"]["min"].get<std::vector<double>>();
  sceneBounds.max = root["sceneBounds"]["max"].get<std::vector<double>>();

  voxelResolution = root["voxelResolution"];
  leafSize = root["leafSize"];
  treeDepth = root["treeDepth"];
  numInteriorNodes = root["numInteriorNodes"];
  numMixedLeaves = root["numMixedLeaves"];
  nodeCount = root["nodeCount"];
  leafDataCount = root["leafDataCount"];
}

std::string VoxelMetadata::dump() const {
  nlohmann::json root;
  root["gridBounds"]["min"] = gridBounds.min;
  root["gridBounds"]["max"] = gridBounds.max;

  root["sceneBounds"]["min"] = sceneBounds.min;
  root["sceneBounds"]["max"] = sceneBounds.max;

  root["voxelResolution"] = voxelResolution;
  root["leafSize"] = leafSize;
  root["treeDepth"] = treeDepth;
  root["numInteriorNodes"] = numInteriorNodes;
  root["numMixedLeaves"] = numMixedLeaves;
  root["nodeCount"] = nodeCount;
  root["leafDataCount"] = leafDataCount;

  return root.dump();
}

}  // namespace splat
