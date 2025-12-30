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

#include <splat/spatial/btree.h>
#include <splat/writers/lod_writer.h>
#include <splat/writers/sog_writer.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace splat {

struct Aabb {
  Eigen::Vector3f min;
  Eigen::Vector3f max;
};

struct MetaLod {
  size_t file;
  size_t offset;
  size_t count;
};

struct MetaNode {
  Aabb bound;
  std::vector<MetaNode> children;  // optional
  std::map<float, MetaLod> lods;
};

struct LodMeta {
  size_t lodLevels;
  std::optional<std::string> environment;
  std::vector<std::string> filenames;
  MetaNode tree;
};

static void boundUnion(Aabb& result, const Aabb& a, const Aabb& b) {
  result.min = a.min.cwiseMin(b.min);
  result.max = a.max.cwiseMax(b.max);
}

static Aabb calcBound(const DataTable* dataTable, const std::vector<uint32_t>& indices) {
  // 1. Get references to columns to avoid massive memory copying
  // Ensure .as<float>() returns a const reference: const std::vector<float>&
  const auto& x = dataTable->getColumnByName("x").asSpan<float>();
  const auto& y = dataTable->getColumnByName("y").asSpan<float>();
  const auto& z = dataTable->getColumnByName("z").asSpan<float>();
  const auto& rx = dataTable->getColumnByName("rot_1").asSpan<float>();
  const auto& ry = dataTable->getColumnByName("rot_2").asSpan<float>();
  const auto& rz = dataTable->getColumnByName("rot_3").asSpan<float>();
  const auto& rw = dataTable->getColumnByName("rot_0").asSpan<float>();
  const auto& sx = dataTable->getColumnByName("scale_0").asSpan<float>();
  const auto& sy = dataTable->getColumnByName("scale_1").asSpan<float>();
  const auto& sz = dataTable->getColumnByName("scale_2").asSpan<float>();

  // Initialize overall bounding box with infinity
  Eigen::Vector3f overallMin(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity());
  Eigen::Vector3f overallMax(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity());

  for (uint32_t index : indices) {
    Eigen::Vector3f p(x[index], y[index], z[index]);

    Eigen::Quaternionf r(rw[index], rx[index], ry[index], rz[index]);
    r.normalize();

    Eigen::Vector3f s(std::exp(sx[index]), std::exp(sy[index]), std::exp(sz[index]));

    Eigen::Matrix3f rotMat = r.toRotationMatrix();

    for (int i = 0; i < 8; ++i) {
      Eigen::Vector3f corner;
      corner << (i & 1 ? s.x() : -s.x()), (i & 2 ? s.y() : -s.y()), (i & 4 ? s.z() : -s.z());

      Eigen::Vector3f transformed = rotMat * corner + p;

      if (transformed.array().isFinite().all()) {
        overallMin = overallMin.cwiseMin(transformed);
        overallMax = overallMax.cwiseMax(transformed);
      }
    }
  }

  return {overallMin, overallMax};
}

static std::map<float, std::vector<uint32_t>> binIndices(BTreeNode* parent, absl::Span<const float> lod) {
  std::map<float, std::vector<uint32_t>> result;

  std::function<void(BTreeNode*)> recurse = [&](BTreeNode* node) {
    if (!node->indices.empty()) {
      for (size_t i = 0; i < node->indices.size(); i++) {
        const auto v = node->indices[i];
        const auto lodValue = lod[v];

        if (result.count(lodValue)) {
          result.insert({lodValue, {v}});
        } else {
          result[lodValue].push_back(v);
        }
      }
    } else {
      if (node->left) {
        recurse(node->left.get());
      }
      if (node->right) {
        recurse(node->right.get());
      }
    }
  };

  recurse(parent);
  return result;
}

void writeLod(const std::string& filename, const DataTable* dataTable, DataTable* envDataTable, bool bundle,
              int iterations, size_t lodChunkCount, size_t lodChunkExtent) {
  fs::path outputDir = fs::path(filename).parent_path();

  // ensure top-level output folder exists
  bool rt = fs::create_directories(outputDir);

  // write the environment sog
  if (envDataTable && envDataTable->getNumRows() > 0) {
    fs::path pathname = outputDir / "env" / "meta.json";
    fs::create_directories(pathname.parent_path());
    std::cout << "writing " << pathname.string() << "..." << std::endl;
    writeSog(pathname.string(), envDataTable, bundle, iterations);
  }

  // construct a kd-tree based on centroids from all lods
  auto centroidsTable = dataTable->clone({"x", "y", "z"});

  BTree btree(centroidsTable.release());
  const size_t binSize = lodChunkCount * 1024;
  const int binDim = lodChunkExtent;

  std::map<float, std::vector<std::vector<std::vector<uint32_t>>>> lodFiles;
  const auto& lodColumn = dataTable->getColumnByName("lod").asSpan<float>();
  std::vector<std::string> filenames;
  float lodLevels = 0;

  std::function<MetaNode(BTreeNode*)> build = [&](BTreeNode* node) -> MetaNode {
    if (node->indices.empty() && (node->count > binSize || node->aabb.largestDim() > binDim)) {
      MetaNode mNode;
      mNode.children.push_back(build(node->left.get()));
      mNode.children.push_back(build(node->right.get()));

      mNode.bound.min.setZero();
      mNode.bound.max.setZero();

      boundUnion(mNode.bound, mNode.children[0].bound, mNode.children[1].bound);

      return mNode;
    }
    std::map<float, MetaLod> lods;
    auto bins = binIndices(node, lodColumn);

    for (auto& [lodValue, indices] : bins) {
      if (!lodFiles.count(lodValue)) {
        lodFiles[lodValue] = {{}};
      }

      auto& fileList = lodFiles[lodValue];
      const auto fileIndex = fileList.size() - 1;
      auto& lastFile = fileList[fileIndex];
      size_t fileSize =
          std::accumulate(lastFile.begin(), lastFile.end(), size_t(0),
                          [](size_t acc, const std::vector<uint32_t>& curr) { return acc + curr.size(); });

      std::string filename = std::to_string(lodValue) + "_" + std::to_string(fileIndex) + "/meta.json";

      auto it = std::find(filenames.begin(), filenames.end(), filename);
      if (it == filenames.end()) {
        it = filenames.insert(it, filename);
      }
      size_t fileIdxInMeta = std::distance(filenames.begin(), it);

      lods.insert({lodValue, {fileIdxInMeta, fileSize, indices.size()}});

      lastFile.push_back(indices);

      if (fileSize + indices.size() > (size_t)binSize) {
        fileList.push_back({});
      }
      lodLevels = std::max(lodLevels, lodValue + 1);
    }

    std::vector<uint32_t> allIndices;
    for (auto const& [key, val] : bins) {
      allIndices.insert(allIndices.end(), val.begin(), val.end());
    }

    auto bound = calcBound(dataTable, allIndices);

    return {bound, {}, lods};
  };

  MetaNode rootMeta = build(btree.root.get());

  std::function<json(const MetaNode&)> metaToJson = [&](const MetaNode& mNode) -> json {
    json j;
    j["bound"] = {{"min", {mNode.bound.min.x(), mNode.bound.min.y(), mNode.bound.min.z()}},
                  {"max", {mNode.bound.max.x(), mNode.bound.max.y(), mNode.bound.max.z()}}};

    if (!mNode.children.empty()) {
      j["children"] = json::array();
      for (const auto& child : mNode.children) j["children"].push_back(metaToJson(child));
    }

    if (!mNode.lods.empty()) {
      j["lods"] = json::object();
      for (auto const& [lodKey, lodVal] : mNode.lods) {
        j["lods"][std::to_string(lodKey)] = {{"file", lodVal.file}, {"offset", lodVal.offset}, {"count", lodVal.count}};
      }
    }
    return j;
  };

  json meta;
  meta["lodLevels"] = lodLevels;
  if (envDataTable && envDataTable->getNumRows() > 0) {
    meta["environment"] = "env/meta.json";
  }
  meta["filenames"] = filenames;
  meta["tree"] = metaToJson(rootMeta);

  std::ofstream ofs(filename);
  ofs << meta.dump(4);
  ofs.flush();
  ofs.close();

  // write file units
  for (auto&& [lodValue, fileUnits] : lodFiles) {
    for (size_t i = 0; i < fileUnits.size(); i++) {
      auto& fileUnit = fileUnits[i];
      if (fileUnit.empty()) continue;

      fs::path pathname = outputDir / (std::to_string(lodValue) + "_" + std::to_string(i)) / "meta.json";
      fs::create_directories(pathname.parent_path());

      size_t totalIndices =
          std::accumulate(fileUnit.begin(), fileUnit.end(), size_t(0),
                          [](size_t acc, const std::vector<uint32_t>& curr) { return acc + curr.size(); });

      std::vector<uint32_t> indices(totalIndices, 0);
      size_t offset = 0;
      for (size_t j = 0; j < fileUnit.size(); j++) {
        std::copy(fileUnit[j].begin(), fileUnit[j].end(), indices.begin() + offset);
        generateOrdering(dataTable, absl::Span<uint32_t>(&indices[offset], fileUnit[j].size()));
        offset += fileUnit[j].size();
      }

      // construct a new table from the ordered data
      auto&& unitDataTable = dataTable->permuteRows(indices);

      writeSog(pathname.string(), unitDataTable.release(), bundle, iterations);
    }
  }
}

}  // namespace splat
