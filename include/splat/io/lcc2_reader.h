/**
 * @file splat/io/lcc2_reader.h
 * @brief LCC2 data structures and reader API.
 *
 * LCC2 is XGRIDS' second-generation 3DGS container format. It stores a JSON
 * tree of spatial nodes, where each leaf node references splat data in
 * standard formats (PLY, SPZ, SOG) by file index + start/count range.
 */
#pragma once

#include <Eigen/Geometry>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace splat {

class SplatCloud;

struct Lcc2NodeData3dgs {
  int name = -1;  // index into splatFiles[]
  int start = 0;
  int count = 0;
};

struct Lcc2NodeDataMesh {
  int name = -1;  // index into meshFiles[]
  int vertex = 0;
  int face = 0;
};

struct Lcc2Node {
  std::string id;
  Eigen::AlignedBox3f boundingBox;
  int childNum = 0;
  std::vector<std::unique_ptr<Lcc2Node>> children;

  std::optional<Lcc2NodeData3dgs> d3dgs;
  std::optional<Lcc2NodeDataMesh> dmesh;

  bool isLeaf() const { return childNum == 0; }
};

struct Lcc2Scene {
  std::string version;
  std::string name;
  std::string description;
  std::string guid;
  std::string fileType;
  int totalSplats = 0;
  int totalLevels = 0;
  std::vector<int> lodSplats;
  std::vector<std::string> splatFiles;
  std::unique_ptr<Lcc2Node> root;

  mutable std::map<int, std::unique_ptr<SplatCloud>> loadedData;
};

Lcc2Scene readLcc2(const std::filesystem::path& lcc2Path);

const SplatCloud* loadNodeData(const Lcc2Scene& scene, const Lcc2Node& node, const std::filesystem::path& baseDir);

std::unique_ptr<SplatCloud> flattenLcc2Scene(const Lcc2Scene& scene, const std::filesystem::path& baseDir);

}  // namespace splat
