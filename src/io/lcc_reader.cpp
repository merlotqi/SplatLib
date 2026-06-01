#include <splat/io/lcc_reader.h>
#include <splat/models/lcc.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace splat {

const float kSH_C0 = 0.28209479177387814f;
const float SQRT_2 = 1.41421356237f;
const float SQRT_2_INV = 0.70710678118f;

static float _min_(float minVal, float maxVal, float s) { return (1.0f - s) * minVal + s * maxVal; }

static float invLinearScale(float v) { return std::log(v); }

static float invSigmoid(float v) { return -std::log((1.0f - v) / v); }

static float invSH0ToColor(float v) { return (v - 0.5f) / kSH_C0; }

static Eigen::Vector3f mixVec3(const Eigen::Vector3f& min, const Eigen::Vector3f& max, const Eigen::Vector3f& v) {
  return Eigen::Vector3f(_min_(min.x(), max.x(), v.x()), _min_(min.y(), max.y(), v.y()),
                         _min_(min.z(), max.z(), v.z()));
}

static void decodePacked_11_10_11(Eigen::Vector3f& res, uint32_t enc) {
  res.x() = static_cast<float>(enc & 0x7FF) / 2047.0f;
  res.y() = static_cast<float>((enc >> 11) & 0x3FF) / 1023.0f;
  res.z() = static_cast<float>((enc >> 21) & 0x7FF) / 2047.0f;
}

// Decode 32-bit packed rotation quaternion and write directly to output arrays.
//
// Encoding: 3 x 10-bit components (range-mapped from [-sqrt(0.5), sqrt(0.5)] to [0,1023])
// plus a 2-bit index (d3) indicating which quaternion component was dropped.
//   d3==0 -> w dropped   (d0=x, d1=y, d2=z stored)
//   d3==1 -> x dropped   (d0=w, d1=y, d2=z stored)
//   d3==2 -> y dropped   (d0=w, d1=x, d2=z stored)
//   d3==3 -> z dropped   (d0=w, d1=x, d2=y stored)
//
// Output convention (matches TS decodeRotationInto):
//   rot_0 = w, rot_1 = x, rot_2 = y, rot_3 = z in the DataTable.
static void decodeRotationInto(uint32_t v, float* rot0, float* rot1, float* rot2, float* rot3,
                               size_t idx) {
  float d0 = static_cast<float>(v & 1023) / 1023.0f;
  float d1 = static_cast<float>((v >> 10) & 1023) / 1023.0f;
  float d2 = static_cast<float>((v >> 20) & 1023) / 1023.0f;
  uint32_t d3 = (v >> 30) & 3;

  float qx = d0 * SQRT_2 - SQRT_2_INV;
  float qy = d1 * SQRT_2 - SQRT_2_INV;
  float qz = d2 * SQRT_2 - SQRT_2_INV;
  float sum = std::min(1.0f, qx * qx + qy * qy + qz * qz);
  float qw = std::sqrt(1.0f - sum);

  // Reconstruct quaternion (w,x,y,z) from (qw,qx,qy,qz) based on which
  // component was dropped. The mapping below matches TS exactly.
  if (d3 == 0) {
    rot0[idx] = qz; rot1[idx] = qw; rot2[idx] = qx; rot3[idx] = qy;
  } else if (d3 == 1) {
    rot0[idx] = qz; rot1[idx] = qx; rot2[idx] = qw; rot3[idx] = qy;
  } else if (d3 == 2) {
    rot0[idx] = qz; rot1[idx] = qx; rot2[idx] = qy; rot3[idx] = qw;
  } else {
    rot0[idx] = qw; rot1[idx] = qx; rot2[idx] = qy; rot3[idx] = qz;
  }
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

std::vector<std::unique_ptr<DataTable>> readLcc(const std::filesystem::path& filename,
                                                const std::filesystem::path& sourceName,
                                                const std::vector<int>& options) {
  (void)filename;
  std::ifstream lccFile(sourceName);
  json lccJson = json::parse(lccFile);

  // Match TS logic for SH detection:
  //   - "Portable" files never have spherical harmonics
  //   - "Quality" files always have spherical harmonics
  //   - Unknown/missing fileType: check for "shcoef" attribute as fallback
  //     (matches TS FIXME for early LCC files without fileType field)
  bool hasSH = false;
  if (lccJson.contains("fileType")) {
    if (lccJson["fileType"] == "Portable") {
      hasSH = false;
    } else if (lccJson["fileType"] == "Quality") {
      hasSH = true;
    } else {
      for (auto& attr : lccJson["attributes"]) {
        if (attr["name"] == "shcoef") { hasSH = true; break; }
      }
    }
  } else {
    hasSH = true;
  }
  CompressInfo compressInfo = parseMeta(lccJson);
  std::vector<int> splats = lccJson["splats"].get<std::vector<int>>();

  const std::filesystem::path baseDir = sourceName.parent_path();
  const std::filesystem::path indexPath = baseDir / "index.bin";
  std::ifstream indexFile(indexPath, std::ios::binary | std::ios::ate);
  std::streamsize idxSize = indexFile.tellg();
  indexFile.seekg(0);
  std::vector<uint8_t> indexData(idxSize);
  indexFile.read(reinterpret_cast<char*>(indexData.data()), idxSize);

  std::ifstream dataFile(baseDir / "data.bin", std::ios::binary);
  std::ifstream shFile;
  if (hasSH) shFile.open(baseDir / "shcoef.bin", std::ios::binary);

  auto unitInfos = parseIndexBin(indexData, lccJson);

  std::vector<std::unique_ptr<DataTable>> result;

  return result;
}

}  // namespace splat
