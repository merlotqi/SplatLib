#include <splat/io/lcc2_reader.h>
#include <splat/io/ply_reader.h>
#include <splat/io/sog_reader.h>
#include <splat/io/spz_reader.h>
#include <splat/models/splatcloud.h>

#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

namespace splat {

namespace {

Eigen::AlignedBox3f parseBoundingBox(const json& bb) {
  Eigen::AlignedBox3f box;
  box.min() = Eigen::Vector3f(bb["min"][0].get<float>(), bb["min"][1].get<float>(), bb["min"][2].get<float>());
  box.max() = Eigen::Vector3f(bb["max"][0].get<float>(), bb["max"][1].get<float>(), bb["max"][2].get<float>());
  return box;
}

std::unique_ptr<Lcc2Node> parseNode(const json& j) {
  auto node = std::make_unique<Lcc2Node>();
  node->id = j["id"].get<std::string>();
  node->boundingBox = parseBoundingBox(j["boundingBox"]);
  node->childNum = j.value("childNum", 0);

  if (node->childNum > 0 && j.contains("child")) {
    for (auto& [key, childJ] : j["child"].items()) {
      node->children.push_back(parseNode(childJ));
    }
  } else if (j.contains("data")) {
    auto& data = j["data"];
    if (data.contains("3dgs")) {
      auto& d = data["3dgs"];
      node->d3dgs = Lcc2NodeData3dgs{d.value("name", -1), d.value("start", 0), d.value("count", 0)};
    }
    if (data.contains("mesh")) {
      auto& m = data["mesh"];
      node->dmesh = Lcc2NodeDataMesh{m.value("name", -1), m.value("vertex", 0), m.value("face", 0)};
    }
  }
  return node;
}

}  // namespace

Lcc2Scene readLcc2(const std::filesystem::path& lcc2Path) {
  std::ifstream f(lcc2Path);
  if (!f) throw std::runtime_error("Cannot open: " + lcc2Path.u8string());

  json j = json::parse(f);

  Lcc2Scene scene;
  scene.version = j.value("version", "0.0.0");
  scene.name = j.value("name", "");
  scene.description = j.value("description", "");
  scene.guid = j.value("guid", "");
  scene.fileType = j.value("fileType", "Quality");
  scene.totalSplats = j.value("totalSplats", 0);
  scene.totalLevels = j.value("totalLevels", 1);

  if (j.contains("lodSplats")) {
    for (auto& v : j["lodSplats"]) scene.lodSplats.push_back(v.get<int>());
  }

  if (j.contains("splatFiles")) {
    for (auto& sf : j["splatFiles"]) scene.splatFiles.push_back(sf.get<std::string>());
  }

  scene.root = parseNode(j["root"]);
  return scene;
}

const SplatCloud* loadNodeData(const Lcc2Scene& scene, const Lcc2Node& node, const std::filesystem::path& baseDir) {
  if (!node.d3dgs || node.d3dgs->name < 0) return nullptr;

  int fileIdx = node.d3dgs->name;
  if (fileIdx < 0 || fileIdx >= static_cast<int>(scene.splatFiles.size())) {
    throw std::runtime_error("LCC2: splatFiles index out of range: " + std::to_string(fileIdx));
  }

  auto it = scene.loadedData.find(fileIdx);
  if (it != scene.loadedData.end()) return it->second.get();

  auto fullPath = baseDir / scene.splatFiles[fileIdx];
  auto ext = fullPath.extension().string();

  std::unique_ptr<SplatCloud> table;
  if (ext == ".ply")
    table = readPly(fullPath);
  else if (ext == ".spz")
    table = readSpz(fullPath);
  else if (ext == ".sog")
    table = readSog(fullPath, fullPath);
  else
    throw std::runtime_error("LCC2: unknown data format: " + ext);

  scene.loadedData[fileIdx] = std::move(table);
  return scene.loadedData[fileIdx].get();
}

std::unique_ptr<SplatCloud> flattenLcc2Scene(const Lcc2Scene& scene, const std::filesystem::path& baseDir) {
  std::vector<const SplatCloud*> parts;
  std::vector<std::vector<uint32_t>> slices;

  std::function<void(const Lcc2Node&)> collect = [&](const Lcc2Node& node) {
    if (node.isLeaf() && node.d3dgs && node.d3dgs->count > 0) {
      const SplatCloud* full = loadNodeData(scene, node, baseDir);
      if (!full) return;

      int start = node.d3dgs->start;
      int count = node.d3dgs->count;

      std::vector<uint32_t> indices;
      indices.reserve(count);
      for (int i = 0; i < count; ++i) indices.push_back(static_cast<uint32_t>(start + i));

      parts.push_back(full);
      slices.push_back(std::move(indices));
    }
    for (auto& child : node.children) collect(*child);
  };
  collect(*scene.root);

  if (parts.empty()) return std::make_unique<SplatCloud>();

  // Build result by permuting each part and concatenating
  std::vector<std::unique_ptr<SplatCloud>> permuted;
  size_t totalRows = 0;
  for (size_t p = 0; p < parts.size(); ++p) {
    permuted.push_back(parts[p]->permuteRows(slices[p]));
    totalRows += slices[p].size();
  }

  // Concatenate: clone first, then add rows from rest
  auto result = permuted[0]->clone();
  for (size_t p = 1; p < permuted.size(); ++p) {
    size_t baseRows = result->getNumRows();
    size_t addRows = permuted[p]->getNumRows();

    // Extend each column in result with data from permuted[p]
    for (size_t ci = 0; ci < permuted[p]->getNumColumns(); ++ci) {
      const auto& srcCol = permuted[p]->getColumn(ci);
      const std::string& colName = srcCol.name;
      if (!result->hasColumn(colName)) continue;

      auto& dstCol = result->getColumnByName(colName);
      for (size_t ri = 0; ri < addRows; ++ri) {
        dstCol.setValue(baseRows + ri, srcCol.getValue<float>(ri));
      }
    }
  }

  return result;
}

}  // namespace splat
