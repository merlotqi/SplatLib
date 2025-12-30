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

#include <splat/models/lcc.h>
#include <splat/readers/lcc_reader.h>

#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace splat {

const float kSH_C0 = 0.28209479177387814f;
const float SQRT_2 = 1.41421356237f;
const float SQRT_2_INV = 0.70710678118f;

static float _min_(float minVal, float maxVal, float s) { return (1.0f - s) * minVal + s * maxVal; }

static float invSigmoid(float v) { return -std::log((1.0f - v) / v); }

static float invSH0ToColor(float v) { return (v - 0.5f) / kSH_C0; }

static Eigen::Vector3f mixVec3(const Eigen::Vector3f& min, const Eigen::Vector3f& max, const Eigen::Vector3f& v) {
  return Eigen::Vector3f(_min_(min.x(), max.x(), v.x()), _min_(min.y(), max.y(), v.y()),
                         _min_(min.z(), max.z(), v.z()));
}

static void decodePacked_11_10_11(Eigen::Vector3f& res, uint32_t enc) {
  res.x() = (enc & 0x7FF) / 2047.0f;
  res.y() = ((enc >> 11) & 0x3FF) / 1023.0f;
  res.z() = ((enc >> 21) & 0x7FF) / 2047.0f;
}

static Eigen::Quaternionf decodeRotation(uint32_t v) {
  float d0 = (v & 1023) / 1023.0f;
  float d1 = ((v >> 10) & 1023) / 1023.0f;
  float d2 = ((v >> 20) & 1023) / 1023.0f;
  uint32_t d3 = (v >> 30) & 3;

  float qx = d0 * SQRT_2 - SQRT_2_INV;
  float qy = d1 * SQRT_2 - SQRT_2_INV;
  float qz = d2 * SQRT_2 - SQRT_2_INV;
  float sum = std::min(1.0f, qx * qx + qy * qy + qz * qz);
  float qw = std::sqrt(1.0f - sum);

  Eigen::Quaternionf q;

  if (d3 == 0) {
    q.w() = qw;
    q.x() = qx;
    q.y() = qy;
    q.z() = qz;
  } else if (d3 == 1) {
    q.w() = qx;
    q.x() = qw;
    q.y() = qy;
    q.z() = qz;
  } else if (d3 == 2) {
    q.w() = qx;
    q.x() = qy;
    q.y() = qw;
    q.z() = qz;
  } else {
    q.w() = qx;
    q.x() = qy;
    q.y() = qz;
    q.z() = qw;
  }
  return q;
}

static std::vector<std::string> floatProps = {"x",       "y",      "z",       "nx",      "ny",     "nz",
                                              "opacity", "rot_0",  "rot_1",   "rot_2",   "rot_3",  "f_dc_0",
                                              "f_dc_1",  "f_dc_2", "scale_0", "scale_1", "scale_2"};

static CompressInfo parseMeta(const json& obj) {
  std::map<std::string, json> attributes;
  for (auto&& attr : obj["attributes"]) {
    attributes[attr["name"].get<std::string>()] = attr;
  }

  auto v3f = [](const json& j) { return Eigen::Vector3f(j[0], j[1], j[2]); };
  CompressInfo info;
  info.scaleMin = v3f(attributes["scale"]["min"]);
  info.scaleMax = v3f(attributes["scale"]["max"]);
  info.shMin = v3f(attributes["shcoef"]["min"]);
  info.shMax = v3f(attributes["shcoef"]["max"]);

  info.envScaleMin = attributes.count("envscale") ? v3f(attributes["envscale"]["min"]) : info.scaleMin;
  info.envScaleMax = attributes.count("envscale") ? v3f(attributes["envscale"]["max"]) : info.scaleMax;
  info.envShMin = attributes.count("envshcoef") ? v3f(attributes["envshcoef"]["min"]) : info.shMin;
  info.envShMax = attributes.count("envshcoef") ? v3f(attributes["envshcoef"]["max"]) : info.shMax;

  return info;
}

static std::vector<LccUnitInfo> parseIndexBin(const std::vector<uint8_t>& raw, const json& meta) {
  size_t offset = 0;
  std::vector<LccUnitInfo> infos;
  int totalLevel = meta["totalLevel"].get<int>();

  while (offset + 4 <= raw.size()) {
    LccUnitInfo info;
    std::memcpy(&info.x, &raw[offset], 2);
    offset += 2;
    std::memcpy(&info.y, &raw[offset], 2);
    offset += 2;

    for (int i = 0; i < totalLevel; i++) {
      LccLod lod = {};
      std::memcpy(&lod.points, &raw[offset], 4);
      offset += 4;
      std::memcpy(&lod.offset, &raw[offset], 8);
      offset += 8;
      std::memcpy(&lod.size, &raw[offset], 4);
      offset += 4;
      info.lods.push_back(lod);
    }
    infos.push_back(info);
  }
  return infos;
}

std::vector<std::unique_ptr<DataTable>> readLcc(const std::string& filename, const std::string& sourceName,
                                                const std::vector<int>& options) {
  std::ifstream lccFile(sourceName);
  json lccJson = json::parse(lccFile);

  bool hasSH = false;
  if (lccJson.contains("fileType")) {
    hasSH = (lccJson["fileType"] == "Quality");
  } else {
    for (auto& attr : lccJson["attributes"])
      if (attr["name"] == "shcoef") hasSH = true;
  }
  CompressInfo compressInfo = parseMeta(lccJson);
  std::vector<int> splats = lccJson["splats"].get<std::vector<int>>();

  std::string baseDir = sourceName.substr(0, sourceName.find_last_of("/\\") + 1);
  std::ifstream indexFile(baseDir + "index.bin", std::ios::binary | std::ios::ate);
  std::streamsize idxSize = indexFile.tellg();
  indexFile.seekg(0);
  std::vector<uint8_t> indexData(idxSize);
  indexFile.read(reinterpret_cast<char*>(indexData.data()), idxSize);

  std::ifstream dataFile(baseDir + "data.bin", std::ios::binary);
  std::ifstream shFile;
  if (hasSH) shFile.open(baseDir + "shcoef.bin", std::ios::binary);

  auto unitInfos = parseIndexBin(indexData, lccJson);

  std::vector<std::unique_ptr<DataTable>> result;

  return result;
}

}  // namespace splat
