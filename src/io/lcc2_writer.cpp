#include <splat/io/lcc2_writer.h>

#include <Eigen/Dense>
#include <splat/io/spz_writer.h>
#include <splat/models/data-table.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace splat {

namespace {

std::string generateGuid() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 15);
  std::stringstream ss;
  for (int i = 0; i < 32; ++i) ss << std::hex << dis(gen);
  return ss.str();
}

void computeBoundingBox(const DataTable& table, Eigen::Vector3f& bboxMin,
                        Eigen::Vector3f& bboxMax) {
  auto xs = table.getColumnByName("x").asSpan<float>();
  auto ys = table.getColumnByName("y").asSpan<float>();
  auto zs = table.getColumnByName("z").asSpan<float>();
  size_t n = table.getNumRows();
  float fmax = std::numeric_limits<float>::max();
  float fmin = std::numeric_limits<float>::lowest();
  bboxMin = Eigen::Vector3f(fmax, fmax, fmax);
  bboxMax = Eigen::Vector3f(fmin, fmin, fmin);
  for (size_t i = 0; i < n; ++i) {
    bboxMin = bboxMin.cwiseMin(Eigen::Vector3f(xs[i], ys[i], zs[i]));
    bboxMax = bboxMax.cwiseMax(Eigen::Vector3f(xs[i], ys[i], zs[i]));
  }
}

std::map<uint32_t, std::vector<size_t>> buildSpatialGrid(const DataTable& table,
                                                          float cellSizeX, float cellSizeY,
                                                          const Eigen::Vector3f& bboxMin) {
  auto xs = table.getColumnByName("x").asSpan<float>();
  auto ys = table.getColumnByName("y").asSpan<float>();
  size_t n = table.getNumRows();
  std::map<uint32_t, std::vector<size_t>> grid;
  for (size_t i = 0; i < n; ++i) {
    int cx = std::clamp(static_cast<int>(std::floor((xs[i] - bboxMin.x()) / cellSizeX)), 0, 65535);
    int cy = std::clamp(static_cast<int>(std::floor((ys[i] - bboxMin.y()) / cellSizeY)), 0, 65535);
    grid[(static_cast<uint32_t>(cy) << 16) | static_cast<uint32_t>(cx)].push_back(i);
  }
  return grid;
}

std::string writeCellFile(const fs::path& dataDir, int fileIdx, const DataTable& table,
                           const std::vector<size_t>& indices) {
  std::vector<uint32_t> u32(indices.begin(), indices.end());
  auto subset = table.permuteRows(u32);

  std::string fname = std::to_string(fileIdx) + ".spz";
  auto fullPath = dataDir / fname;
  writeSpz(fullPath, *subset);
  return "data/3dgs/" + fname;
}

}  // namespace

void writeLcc2(const fs::path& outputDir, const std::vector<const DataTable*>& lods,
               const Lcc2WriteConfig& config) {
  if (lods.empty()) throw std::runtime_error("LCC2 writer: at least one LOD required");

  fs::create_directories(outputDir);
  auto dataDir = outputDir / "data" / "3dgs";
  fs::create_directories(dataDir);

  size_t numLods = lods.size();

  // Build spatial grids per LOD
  std::vector<std::map<uint32_t, std::vector<size_t>>> grids(numLods);
  std::set<uint32_t> allCellIds;
  Eigen::Vector3f globalMin, globalMax;
  computeBoundingBox(*lods[0], globalMin, globalMax);

  for (size_t lod = 0; lod < numLods; ++lod) {
    grids[lod] = buildSpatialGrid(*lods[lod], config.cellSizeX, config.cellSizeY, globalMin);
    for (auto& kv : grids[lod]) allCellIds.insert(kv.first);
  }

  // Encode cells
  struct CellEntry {
    uint32_t cellId;
    size_t lod;
    int fileIdx;
    int splatCount;
    Eigen::AlignedBox3f bbox;
  };
  std::vector<CellEntry> entries;
  std::vector<std::string> splatFiles;
  std::vector<size_t> splatsPerLod(numLods, 0);
  int fileIdx = 0;

  for (uint32_t cellId : allCellIds) {
    for (size_t lod = 0; lod < numLods; ++lod) {
      auto it = grids[lod].find(cellId);
      if (it == grids[lod].end() || it->second.empty()) continue;

      auto& xs = lods[lod]->getColumnByName("x").asSpan<float>();
      auto& ys = lods[lod]->getColumnByName("y").asSpan<float>();
      auto& zs = lods[lod]->getColumnByName("z").asSpan<float>();
      Eigen::AlignedBox3f cellBBox;
      cellBBox.min() = Eigen::Vector3f(xs[it->second[0]], ys[it->second[0]], zs[it->second[0]]);
      cellBBox.max() = cellBBox.min();
      for (auto idx : it->second)
        cellBBox.extend(Eigen::Vector3f(xs[idx], ys[idx], zs[idx]));

      std::string rel = writeCellFile(dataDir, fileIdx, *lods[lod], it->second);
      splatFiles.push_back(rel);

      entries.push_back(
          {cellId, lod, fileIdx, static_cast<int>(it->second.size()), cellBBox});
      splatsPerLod[lod] += it->second.size();
      fileIdx++;
    }
  }

  size_t totalSplats = 0;
  for (auto c : splatsPerLod) totalSplats += c;

  // Build tree: root → per-cell internal nodes → per-LOD leaves
  std::vector<json> cellNodes;
  size_t e = 0;
  for (uint32_t cellId : allCellIds) {
    int cx = cellId & 0xFFFF, cy = (cellId >> 16) & 0xFFFF;
    std::string cellIdStr = std::to_string(cx) + "-" + std::to_string(cy);

    Eigen::AlignedBox3f mergedBox;
    bool first = true;
    std::vector<json> lodLeaves;
    while (e < entries.size() && entries[e].cellId == cellId) {
      auto& ent = entries[e];
      if (first) {
        mergedBox = ent.bbox;
        first = false;
      } else {
        mergedBox.extend(ent.bbox);
      }

      json leaf;
      leaf["id"] = "0-" + cellIdStr + "-" + std::to_string(ent.lod);
      leaf["boundingBox"]["min"] = {ent.bbox.min().x(), ent.bbox.min().y(), ent.bbox.min().z()};
      leaf["boundingBox"]["max"] = {ent.bbox.max().x(), ent.bbox.max().y(), ent.bbox.max().z()};
      leaf["childNum"] = 0;
      leaf["data"]["3dgs"]["name"] = ent.fileIdx;
      leaf["data"]["3dgs"]["start"] = 0;
      leaf["data"]["3dgs"]["count"] = ent.splatCount;
      lodLeaves.push_back(leaf);
      e++;
    }

    json cellNode;
    cellNode["id"] = "0-" + cellIdStr;
    cellNode["boundingBox"]["min"] = {mergedBox.min().x(), mergedBox.min().y(),
                                       mergedBox.min().z()};
    cellNode["boundingBox"]["max"] = {mergedBox.max().x(), mergedBox.max().y(),
                                       mergedBox.max().z()};
    cellNode["childNum"] = static_cast<int>(lodLeaves.size());
    json childObj = json::object();
    for (size_t i = 0; i < lodLeaves.size(); ++i)
      childObj[std::to_string(i)] = lodLeaves[i];
    cellNode["child"] = childObj;
    cellNodes.push_back(cellNode);
  }

  // Root
  json root;
  root["id"] = "0";
  root["boundingBox"]["min"] = {globalMin.x(), globalMin.y(), globalMin.z()};
  root["boundingBox"]["max"] = {globalMax.x(), globalMax.y(), globalMax.z()};
  root["childNum"] = static_cast<int>(cellNodes.size());
  json rootChild = json::object();
  for (size_t i = 0; i < cellNodes.size(); ++i)
    rootChild[std::to_string(i)] = cellNodes[i];
  root["child"] = rootChild;

  // Scene JSON
  std::string guid = config.guid.empty() ? generateGuid() : config.guid;
  json scene;
  scene["version"] = "0.0.3";
  scene["name"] = config.name;
  scene["description"] = config.description;
  scene["guid"] = guid;
  scene["fileType"] = config.fileType;
  scene["totalSplats"] = totalSplats;
  scene["totalLevels"] = numLods;

  json lodArr = json::array();
  for (auto c : splatsPerLod) lodArr.push_back(static_cast<int>(c));
  scene["lodSplats"] = lodArr;
  scene["splatFiles"] = splatFiles;
  scene["root"] = root;
  scene["source"] = "lcc";
  scene["dataType"] = "DIMENVUE";
  scene["epsg"] = 0;
  scene["splatType"] = "." + config.outputFormat;

  std::ofstream out(outputDir / (config.name + ".lcc2"));
  if (!out) throw std::runtime_error("Failed to create scene.lcc2");
  out << std::setprecision(15) << scene.dump(1, '\t') << '\n';
}

}  // namespace splat
