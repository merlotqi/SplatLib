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
