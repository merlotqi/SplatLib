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

// Decode one quadtree unit's splat data into pre-allocated property arrays.
//
// Reads unitSplats x 32 bytes from dataFile at lod.offset,
// and optionally unitSplats x 64 bytes from shFile at lod.offset * 2.
//
// f_rest_bands: pointer to first of 45 contiguous std::vector<float>, each
// sized grandTotal. Null if SH data is not present.
//
// Coordinate rotation (LCC -> PlayCanvas): fromEulers(90, 0, 180)
//   Position:  x' = -x,  y' = z,  z' = y
//   Rotation:  q_combined = q_transform * q_splat
//   q_transform = (0, 0, sqrt(0.5), sqrt(0.5))
static void processUnit(const LccUnitInfo& info, int targetLod, std::ifstream& dataFile,
                        std::ifstream* shFile, const CompressInfo& compressInfo,
                        size_t propertyOffset,
                        std::map<std::string, std::vector<float>>& properties,
                        std::vector<float>* f_rest_bands) {
  const auto& lod = info.lods[static_cast<size_t>(targetLod)];
  const int unitSplats = lod.points;
  if (unitSplats == 0) return;

  const int64_t dataOffset = lod.offset;
  const int dataSize = lod.size;

  // Read data.bin range for this unit
  std::vector<uint8_t> dataBytes(static_cast<size_t>(dataSize));
  dataFile.seekg(dataOffset);
  dataFile.read(reinterpret_cast<char*>(dataBytes.data()), dataSize);

  // Read shcoef.bin range if present (offset x 2: SH stride is 64 vs data 32)
  std::vector<uint8_t> shBytes;
  if (shFile && shFile->is_open()) {
    size_t shSize = static_cast<size_t>(unitSplats) * 64;
    shBytes.resize(shSize);
    shFile->seekg(dataOffset * 2);
    shFile->read(reinterpret_cast<char*>(shBytes.data()),
                 static_cast<std::streamsize>(shSize));
  }

  // Extract array references once
  auto& px = properties["property_x"];
  auto& py = properties["property_y"];
  auto& pz = properties["property_z"];
  auto& pnx = properties["property_nx"];
  auto& pny = properties["property_ny"];
  auto& pnz = properties["property_nz"];
  auto& pop = properties["property_opacity"];
  auto& pr0 = properties["property_rot_0"];
  auto& pr1 = properties["property_rot_1"];
  auto& pr2 = properties["property_rot_2"];
  auto& pr3 = properties["property_rot_3"];
  auto& pdc0 = properties["property_f_dc_0"];
  auto& pdc1 = properties["property_f_dc_1"];
  auto& pdc2 = properties["property_f_dc_2"];
  auto& ps0 = properties["property_scale_0"];
  auto& ps1 = properties["property_scale_1"];
  auto& ps2 = properties["property_scale_2"];

  const float sMinX = compressInfo.scaleMin.x(), sMinY = compressInfo.scaleMin.y(),
              sMinZ = compressInfo.scaleMin.z();
  const float sMaxX = compressInfo.scaleMax.x(), sMaxY = compressInfo.scaleMax.y(),
              sMaxZ = compressInfo.scaleMax.z();
  const float shMinX = compressInfo.shMin.x(), shMinY = compressInfo.shMin.y(),
              shMinZ = compressInfo.shMin.z();
  const float shMaxX = compressInfo.shMax.x(), shMaxY = compressInfo.shMax.y(),
              shMaxZ = compressInfo.shMax.z();

  // LCC->PlayCanvas coordinate transform quaternion: fromEulers(90, 0, 180)
  // q_transform = (0, 0, sqrt(0.5), sqrt(0.5))
  const float qtw = 0.0f, qtx = 0.0f, qty = 0.70710678118f, qtz = 0.70710678118f;

  const float* f32view = reinterpret_cast<const float*>(dataBytes.data());
  const uint16_t* u16view = reinterpret_cast<const uint16_t*>(dataBytes.data());
  const uint8_t* u8data = dataBytes.data();

  for (int i = 0; i < unitSplats; ++i) {
    const size_t g = propertyOffset + static_cast<size_t>(i);
    const size_t fi = static_cast<size_t>(i) << 3;   // i * 8 (f32 index per 32-byte record)
    const size_t bi = static_cast<size_t>(i) << 5;   // i * 32 (byte index)
    const size_t hi = static_cast<size_t>(i) << 4;   // i * 16 (u16 index per 32-byte record)

    // Position: f32 at dataBytes[0..11]
    float pos_x = f32view[fi];
    float pos_y = f32view[fi + 1];
    float pos_z = f32view[fi + 2];

    // Color + opacity: u8 at dataBytes[12..15]
    pdc0[g] = invSH0ToColor(static_cast<float>(u8data[bi + 12]) / 255.0f);
    pdc1[g] = invSH0ToColor(static_cast<float>(u8data[bi + 13]) / 255.0f);
    pdc2[g] = invSH0ToColor(static_cast<float>(u8data[bi + 14]) / 255.0f);
    pop[g] = invSigmoid(static_cast<float>(u8data[bi + 15]) / 255.0f);

    // Scale: u16 at dataBytes[16..21], dequantize -> linear -> log
    ps0[g] = invLinearScale(
        _min_(sMinX, sMaxX, static_cast<float>(u16view[hi + 8]) / 65535.0f));
    ps1[g] = invLinearScale(
        _min_(sMinY, sMaxY, static_cast<float>(u16view[hi + 9]) / 65535.0f));
    ps2[g] = invLinearScale(
        _min_(sMinZ, sMaxZ, static_cast<float>(u16view[hi + 10]) / 65535.0f));

    // Rotation: u16[11] | (u16[12] << 16) -> packed u32 from dataBytes[22..25]
    uint32_t rotEnc = static_cast<uint32_t>(u16view[hi + 11]) |
                      (static_cast<uint32_t>(u16view[hi + 12]) << 16);
    decodeRotationInto(rotEnc, pr0.data(), pr1.data(), pr2.data(), pr3.data(), g);

    // Apply coordinate rotation to quaternion: q_combined = q_transform * q_splat
    float qsw = pr0[g], qsx = pr1[g], qsy = pr2[g], qsz = pr3[g];
    float cw = qtw * qsw - qtx * qsx - qty * qsy - qtz * qsz;
    float cx = qtw * qsx + qtx * qsw + qty * qsz - qtz * qsy;
    float cy = qtw * qsy - qtx * qsz + qty * qsw + qtz * qsx;
    float cz = qtw * qsz + qtx * qsy - qty * qsx + qtz * qsw;
    if (cw < 0.0f) { cw = -cw; cx = -cx; cy = -cy; cz = -cz; }
    pr0[g] = cw; pr1[g] = cx; pr2[g] = cy; pr3[g] = cz;

    // Apply coordinate rotation to position: x'=-x, y'=z, z'=y
    px[g] = -pos_x;
    py[g] = pos_z;
    pz[g] = pos_y;

    // Normals: u16 at dataBytes[26..31], stored raw
    pnx[g] = static_cast<float>(u16view[hi + 13]);
    pny[g] = static_cast<float>(u16view[hi + 14]);
    pnz[g] = static_cast<float>(u16view[hi + 15]);

    // SH coefficients: 15 x u32 per splat (from shcoef.bin, 64-byte stride)
    if (!shBytes.empty() && f_rest_bands) {
      const uint32_t* shU32 = reinterpret_cast<const uint32_t*>(shBytes.data());
      const size_t si = static_cast<size_t>(i) << 4;  // i * 16 (u32 index, 64-byte stride)
      for (int j = 0; j < 15; ++j) {
        uint32_t enc = shU32[si + static_cast<size_t>(j)];
        float nx = static_cast<float>(enc & 0x7FF) / 2047.0f;
        float ny = static_cast<float>((enc >> 11) & 0x3FF) / 1023.0f;
        float nz = static_cast<float>((enc >> 21) & 0x7FF) / 2047.0f;
        f_rest_bands[static_cast<size_t>(j)][g] =
            _min_(shMinX, shMaxX, nx);
        f_rest_bands[static_cast<size_t>(j + 15)][g] =
            _min_(shMinY, shMaxY, ny);
        f_rest_bands[static_cast<size_t>(j + 30)][g] =
            _min_(shMinZ, shMaxZ, nz);
      }
    }
  }
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
