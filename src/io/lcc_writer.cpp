#include <splat/io/lcc_writer.h>
#include <splat/models/data-table.h>
#include <splat/models/lcc.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "lcc_encoder.h"

using json = nlohmann::json;

namespace splat {

namespace {

// ── Helpers ────────────────────────────────────────────────────────────────

/// Generate a random 32-hex-digit GUID string.
std::string generateGuid() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 15);
  std::stringstream ss;
  for (int i = 0; i < 32; ++i) ss << std::hex << dis(gen);
  return ss.str();
}

/// Get a const span of float values from a DataTable column by name.
/// Throws if the column is missing or has wrong row count.
absl::Span<const float> getFloatSpan(const DataTable& table, const std::string& name) {
  const Column& col = table.getColumnByName(name);
  if (col.getType() != ColumnType::FLOAT32) {
    throw std::runtime_error("LCC writer: column '" + name + "' must be float32");
  }
  return col.asSpan<float>();
}

/// Safely get a float span, returning empty span if column doesn't exist.
absl::Span<const float> getOptionalFloatSpan(const DataTable& table, const std::string& name) {
  if (!table.hasColumn(name)) return {};
  return getFloatSpan(table, name);
}

// ── Coordinate transform (inverse of reader's PlayCanvas conversion) ──────

/// Apply the inverse of the reader's fromEulers(90, 0, 180) coordinate
/// transform so that a round-trip (write → read) is identity.
///
/// Position:  (-x, z, y) — self-inverse.
/// Rotation:  multiply by conjugate of fromEulers(90°, 0°, 180°).
void applyInverseCoordinateTransform(float& px, float& py, float& pz, float& rw, float& rx, float& ry, float& rz) {
  // Position: (-x, z, y)
  float tx = -px;
  float ty = pz;
  float tz = py;
  px = tx;
  py = ty;
  pz = tz;

  // Rotation: multiply by qr* = conjugate of (0, h, 0, h)
  // where h = sqrt(2)/2.
  constexpr float h = 0.7071067811865475f;
  float nw = h * (rx + rz);
  float nx = h * (ry - rw);
  float ny = h * (rz - rx);
  float nz = -h * (rw + ry);

  rw = nw;
  rx = nx;
  ry = ny;
  rz = nz;
}

// ── Range collection ──────────────────────────────────────────────────────

/// Scan all LOD DataTables and compute the CompressInfo (min/max ranges)
/// and global bounding box.
struct CollectedRanges {
  CompressInfo compress;
  Eigen::Vector3f bboxMin;
  Eigen::Vector3f bboxMax;
  bool hasSH = false;
};

CollectedRanges collectRanges(const std::vector<const DataTable*>& lods) {
  CollectedRanges r;
  // Initialise to extreme values
  float fmax = std::numeric_limits<float>::max();
  float fmin = std::numeric_limits<float>::lowest();
  r.bboxMin = Eigen::Vector3f(fmax, fmax, fmax);
  r.bboxMax = Eigen::Vector3f(fmin, fmin, fmin);

  r.compress.scaleMin = Eigen::Vector3f(fmax, fmax, fmax);
  r.compress.scaleMax = Eigen::Vector3f(fmin, fmin, fmin);

  float shMin = fmax, shMax = fmin;
  float opMin = fmax, opMax = fmin;

  for (const auto* table : lods) {
    size_t n = table->getNumRows();
    auto xs = getFloatSpan(*table, "x");
    auto ys = getFloatSpan(*table, "y");
    auto zs = getFloatSpan(*table, "z");
    auto s0 = getFloatSpan(*table, "scale_0");
    auto s1 = getFloatSpan(*table, "scale_1");
    auto s2 = getFloatSpan(*table, "scale_2");
    auto op = getFloatSpan(*table, "opacity");

    // Check for SH rest coefficients in the first table
    if (table == lods[0] && table->hasColumn("f_rest_0")) {
      r.hasSH = true;
    }

    for (size_t i = 0; i < n; ++i) {
      float px = xs[i], py = ys[i], pz = zs[i];

      // Bounding box
      r.bboxMin.x() = std::min(r.bboxMin.x(), px);
      r.bboxMin.y() = std::min(r.bboxMin.y(), py);
      r.bboxMin.z() = std::min(r.bboxMin.z(), pz);
      r.bboxMax.x() = std::max(r.bboxMax.x(), px);
      r.bboxMax.y() = std::max(r.bboxMax.y(), py);
      r.bboxMax.z() = std::max(r.bboxMax.z(), pz);

      // Scale (convert from log to linear space)
      float ls0 = std::exp(s0[i]);
      float ls1 = std::exp(s1[i]);
      float ls2 = std::exp(s2[i]);
      r.compress.scaleMin.x() = std::min(r.compress.scaleMin.x(), ls0);
      r.compress.scaleMin.y() = std::min(r.compress.scaleMin.y(), ls1);
      r.compress.scaleMin.z() = std::min(r.compress.scaleMin.z(), ls2);
      r.compress.scaleMax.x() = std::max(r.compress.scaleMax.x(), ls0);
      r.compress.scaleMax.y() = std::max(r.compress.scaleMax.y(), ls1);
      r.compress.scaleMax.z() = std::max(r.compress.scaleMax.z(), ls2);

      // Opacity
      float sigOp = lccSigmoid(op[i]);
      opMin = std::min(opMin, sigOp);
      opMax = std::max(opMax, sigOp);
    }
  }

  // Collect SH ranges (only need from LOD0 for global range)
  if (r.hasSH) {
    const DataTable* t0 = lods[0];
    auto dc0 = getFloatSpan(*t0, "f_dc_0");
    auto dc1 = getFloatSpan(*t0, "f_dc_1");
    auto dc2 = getFloatSpan(*t0, "f_dc_2");
    size_t n = t0->getNumRows();

    for (size_t i = 0; i < n; ++i) {
      shMin = std::min({shMin, dc0[i], dc1[i], dc2[i]});
      shMax = std::max({shMax, dc0[i], dc1[i], dc2[i]});
    }

    // Also scan f_rest for SH range
    for (int b = 0; b < 45; ++b) {
      std::string name = "f_rest_" + std::to_string(b);
      if (!t0->hasColumn(name)) break;
      auto fr = getFloatSpan(*t0, name);
      for (size_t i = 0; i < n; ++i) {
        shMin = std::min(shMin, fr[i]);
        shMax = std::max(shMax, fr[i]);
      }
    }
  }

  r.compress.shMin = Eigen::Vector3f(shMin, shMin, shMin);
  r.compress.shMax = Eigen::Vector3f(shMax, shMax, shMax);

  // Env ranges: use same as splat ranges as defaults (per ply2lcc)
  r.compress.envScaleMin = r.compress.scaleMin;
  r.compress.envScaleMax = r.compress.scaleMax;
  r.compress.envShMin = r.compress.shMin;
  r.compress.envShMax = r.compress.shMax;

  return r;
}

// ── Spatial partitioning ──────────────────────────────────────────────────

/// Per-cell splat index accumulator.
struct CellAccum {
  uint32_t cellId;
  std::vector<size_t> indices;  // splat row indices in the DataTable
};

/// Build the spatial grid: assign each splat to a cell.
std::map<uint32_t, std::vector<size_t>> buildSpatialGrid(const DataTable& table, const Eigen::Vector3f& bboxMin,
                                                         float cellSizeX, float cellSizeY) {
  auto xs = getFloatSpan(table, "x");
  auto ys = getFloatSpan(table, "y");
  size_t n = table.getNumRows();

  std::map<uint32_t, std::vector<size_t>> grid;

  for (size_t i = 0; i < n; ++i) {
    int cx = static_cast<int>(std::floor((xs[i] - bboxMin.x()) / cellSizeX));
    int cy = static_cast<int>(std::floor((ys[i] - bboxMin.y()) / cellSizeY));
    cx = std::max(0, std::min(cx, 65535));
    cy = std::max(0, std::min(cy, 65535));
    uint32_t id = (static_cast<uint32_t>(cy) << 16) | static_cast<uint32_t>(cx);
    grid[id].push_back(i);
  }
  return grid;
}

// ── Per-cell encoding ─────────────────────────────────────────────────────

/// Encoded data for a single cell at a single LOD.
struct EncodedCell {
  uint32_t cellId;
  size_t splatCount;
  std::vector<uint8_t> data;    // 32 bytes × splatCount
  std::vector<uint8_t> shcoef;  // 64 bytes × splatCount (empty if !hasSH)
};

EncodedCell encodeCell(const DataTable& table, const std::vector<size_t>& splatIndices, const CollectedRanges& ranges,
                       const LccWriteConfig& config, uint32_t cellId) {
  EncodedCell cell;
  cell.cellId = cellId;
  cell.splatCount = splatIndices.size();
  cell.data.resize(cell.splatCount * 32);
  if (ranges.hasSH) {
    cell.shcoef.resize(cell.splatCount * 64);
  }

  // Get column spans
  auto xs = getFloatSpan(table, "x");
  auto ys = getFloatSpan(table, "y");
  auto zs = getFloatSpan(table, "z");
  auto r0 = getFloatSpan(table, "rot_0");
  auto r1 = getFloatSpan(table, "rot_1");
  auto r2 = getFloatSpan(table, "rot_2");
  auto r3 = getFloatSpan(table, "rot_3");
  auto dc0 = getFloatSpan(table, "f_dc_0");
  auto dc1 = getFloatSpan(table, "f_dc_1");
  auto dc2 = getFloatSpan(table, "f_dc_2");
  auto sc0 = getFloatSpan(table, "scale_0");
  auto sc1 = getFloatSpan(table, "scale_1");
  auto sc2 = getFloatSpan(table, "scale_2");
  auto op = getFloatSpan(table, "opacity");

  // Optional f_rest columns
  float f_rest[45] = {};
  std::vector<absl::Span<const float>> f_rest_spans;
  if (ranges.hasSH) {
    f_rest_spans.reserve(45);
    for (int b = 0; b < 45; ++b) {
      f_rest_spans.push_back(getOptionalFloatSpan(table, "f_rest_" + std::to_string(b)));
    }
  }

  float sh_min_scalar = ranges.compress.shMin.x();
  float sh_max_scalar = ranges.compress.shMax.x();

  for (size_t j = 0; j < cell.splatCount; ++j) {
    size_t idx = splatIndices[j];

    float pos[3] = {xs[idx], ys[idx], zs[idx]};
    float rot[4] = {r0[idx], r1[idx], r2[idx], r3[idx]};
    float fdc[3] = {dc0[idx], dc1[idx], dc2[idx]};
    float scl[3] = {sc0[idx], sc1[idx], sc2[idx]};
    float opacity = op[idx];

    // Apply inverse coordinate transform if requested
    if (config.applyCoordinateTransform) {
      applyInverseCoordinateTransform(pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]);
    }

    // Collect f_rest for this splat if SH is enabled
    float* fr_ptr = nullptr;
    if (ranges.hasSH) {
      for (int b = 0; b < 45; ++b) {
        f_rest[b] = f_rest_spans[b].empty() ? 0.0f : f_rest_spans[b][idx];
      }
      fr_ptr = f_rest;
    }

    uint8_t* dataPtr = cell.data.data() + j * 32;
    uint8_t* shPtr = ranges.hasSH ? (cell.shcoef.data() + j * 64) : nullptr;

    encodeSplat(pos, fdc, opacity, scl, rot, fr_ptr, ranges.compress.scaleMin, ranges.compress.scaleMax, sh_min_scalar,
                sh_max_scalar, ranges.hasSH, dataPtr, shPtr);
  }

  return cell;
}

// ── File output ───────────────────────────────────────────────────────────

void writeMetaLcc(const std::filesystem::path& path, const CollectedRanges& ranges,
                  const std::vector<size_t>& splatsPerLod, size_t totalSplats, size_t numLods,
                  const LccWriteConfig& config) {
  std::string fileType = ranges.hasSH ? "Quality" : "Portable";
  std::string guid = config.guid.empty() ? generateGuid() : config.guid;
  int indexDataSize = 4 + 16 * static_cast<int>(numLods);

  auto v3 = [](const Eigen::Vector3f& v) -> json { return {v.x(), v.y(), v.z()}; };

  // splats array
  json splatsArr = json::array();
  for (auto c : splatsPerLod) splatsArr.push_back(c);

  // attributes array
  json attrs = json::array();

  // position
  attrs.push_back({{"name", "position"}, {"min", v3(ranges.bboxMin)}, {"max", v3(ranges.bboxMax)}});

  // normal
  attrs.push_back({{"name", "normal"}, {"min", {0, 0, 0}}, {"max", {0, 0, 0}}});

  // color
  attrs.push_back({{"name", "color"}, {"min", {0, 0, 0}}, {"max", {1, 1, 1}}});

  // shcoef
  attrs.push_back({{"name", "shcoef"},
                   {"min", ranges.hasSH ? v3(ranges.compress.shMin) : json({0, 0, 0})},
                   {"max", ranges.hasSH ? v3(ranges.compress.shMax) : json({1, 1, 1})}});

  // opacity
  attrs.push_back({{"name", "opacity"}, {"min", {0.0}}, {"max", {1.0}}});

  // scale
  attrs.push_back({{"name", "scale"}, {"min", v3(ranges.compress.scaleMin)}, {"max", v3(ranges.compress.scaleMax)}});

  // envnormal
  attrs.push_back({{"name", "envnormal"}, {"min", {0, 0, 0}}, {"max", {0, 0, 0}}});

  // envshcoef
  attrs.push_back({{"name", "envshcoef"},
                   {"min", ranges.hasSH ? v3(ranges.compress.envShMin) : json({0, 0, 0})},
                   {"max", ranges.hasSH ? v3(ranges.compress.envShMax) : json({1, 1, 1})}});

  // envscale
  attrs.push_back(
      {{"name", "envscale"}, {"min", v3(ranges.compress.envScaleMin)}, {"max", v3(ranges.compress.envScaleMax)}});

  json meta = {
      {"version", "5.0"},
      {"guid", guid},
      {"name", config.name},
      {"description", config.description},
      {"source", "lcc"},
      {"dataType", "DIMENVUE"},
      {"totalSplats", totalSplats},
      {"totalLevel", numLods},
      {"cellLengthX", config.cellSizeX},
      {"cellLengthY", config.cellSizeY},
      {"indexDataSize", indexDataSize},
      {"offset", {0, 0, 0}},
      {"epsg", 0},
      {"shift", {0, 0, 0}},
      {"scale", {1, 1, 1}},
      {"splats", splatsArr},
      {"boundingBox", {{"min", v3(ranges.bboxMin)}, {"max", v3(ranges.bboxMax)}}},
      {"encoding", "COMPRESS"},
      {"fileType", fileType},
      {"attributes", attrs},
  };

  std::ofstream f(path);
  if (!f) throw std::runtime_error("Failed to create meta.lcc");
  f << std::setprecision(15) << meta.dump(1, '\t') << '\n';
}

}  // anonymous namespace

// ── Public API ─────────────────────────────────────────────────────────────

void writeLcc(const std::filesystem::path& outputDir, const std::vector<const DataTable*>& lods,
              const LccWriteConfig& config) {
  if (lods.empty()) {
    throw std::runtime_error("LCC writer: at least one LOD DataTable required");
  }

  std::filesystem::create_directories(outputDir);

  // Phase 1: collect ranges
  CollectedRanges ranges = collectRanges(lods);
  size_t numLods = lods.size();

  // Phase 2: build spatial grids for all LODs
  // All LODs share the same cell scheme derived from LOD0 bbox
  std::vector<std::map<uint32_t, std::vector<size_t>>> grids;
  grids.reserve(numLods);
  for (size_t lod = 0; lod < numLods; ++lod) {
    grids.push_back(buildSpatialGrid(*lods[lod], ranges.bboxMin, config.cellSizeX, config.cellSizeY));
  }

  // Collect all unique cell IDs across all LODs
  std::set<uint32_t> allCellIds;
  for (const auto& g : grids) {
    for (const auto& kv : g) allCellIds.insert(kv.first);
  }

  // Phase 3: encode cells, tracking LOD per entry

  struct CellLodData {
    uint32_t cellId;
    size_t lod;
    uint32_t splatCount;
    uint64_t dataOffset;
    uint32_t dataSize;
    uint64_t shOffset;
    uint32_t shSize;
    std::vector<uint8_t> dataBuf;
    std::vector<uint8_t> shBuf;
  };

  std::vector<CellLodData> cellLodEntries;
  cellLodEntries.reserve(allCellIds.size() * numLods);

  for (uint32_t cellId : allCellIds) {
    for (size_t lod = 0; lod < numLods; ++lod) {
      auto it = grids[lod].find(cellId);
      if (it == grids[lod].end() || it->second.empty()) continue;

      EncodedCell cell = encodeCell(*lods[lod], it->second, ranges, config, cellId);

      CellLodData entry;
      entry.cellId = cellId;
      entry.lod = lod;
      entry.splatCount = static_cast<uint32_t>(cell.splatCount);
      entry.dataSize = static_cast<uint32_t>(cell.data.size());
      entry.shSize = static_cast<uint32_t>(cell.shcoef.size());
      entry.dataBuf = std::move(cell.data);
      entry.shBuf = std::move(cell.shcoef);
      cellLodEntries.push_back(std::move(entry));
    }
  }

  // Compute offsets by walking entries in cellId→lod order
  uint64_t dataFileOffset = 0;
  uint64_t shFileOffset = 0;

  for (auto& entry : cellLodEntries) {
    entry.dataOffset = dataFileOffset;
    dataFileOffset += entry.dataSize;
    if (ranges.hasSH && entry.shSize > 0) {
      entry.shOffset = shFileOffset;
      shFileOffset += entry.shSize;
    }
  }

  // Write data.bin and accumulate index entries
  std::ofstream dataFile(outputDir / "data.bin", std::ios::binary);
  if (!dataFile) throw std::runtime_error("Failed to create data.bin");

  std::ofstream shFile;
  if (ranges.hasSH) {
    shFile.open(outputDir / "shcoef.bin", std::ios::binary);
    if (!shFile) throw std::runtime_error("Failed to create shcoef.bin");
  }

  // Build index entries grouped by cellId
  struct FinalIndexLod {
    uint32_t points;
    uint64_t offset;
    uint32_t size;
  };
  std::map<uint32_t, std::vector<FinalIndexLod>> finalIndex;  // cellId → per-LOD

  for (auto& entry : cellLodEntries) {
    dataFile.write(reinterpret_cast<const char*>(entry.dataBuf.data()), entry.dataSize);
    if (ranges.hasSH && entry.shSize > 0) {
      shFile.write(reinterpret_cast<const char*>(entry.shBuf.data()), entry.shSize);
    }

    auto& lodVec = finalIndex[entry.cellId];
    if (lodVec.size() < numLods) lodVec.resize(numLods);
    lodVec[entry.lod] = {entry.splatCount, entry.dataOffset, entry.dataSize};
  }

  dataFile.close();
  if (shFile.is_open()) shFile.close();

  // Recompute totalSplats from all entries
  size_t totalSplats = 0;
  std::vector<size_t> splatsPerLod(numLods, 0);
  for (const auto& entry : cellLodEntries) {
    totalSplats += entry.splatCount;
    splatsPerLod[entry.lod] += entry.splatCount;
  }

  // Write index.bin
  {
    std::ofstream idxFile(outputDir / "index.bin", std::ios::binary);
    if (!idxFile) throw std::runtime_error("Failed to create index.bin");

    for (const auto& kv : finalIndex) {
      uint32_t unitIndex = kv.first;
      idxFile.write(reinterpret_cast<const char*>(&unitIndex), 4);

      const auto& lodVec = kv.second;
      for (size_t lod = 0; lod < numLods; ++lod) {
        if (lod < lodVec.size()) {
          idxFile.write(reinterpret_cast<const char*>(&lodVec[lod].points), 4);
          idxFile.write(reinterpret_cast<const char*>(&lodVec[lod].offset), 8);
          idxFile.write(reinterpret_cast<const char*>(&lodVec[lod].size), 4);
        } else {
          // Empty LOD for this cell
          uint32_t zero32 = 0;
          uint64_t zero64 = 0;
          idxFile.write(reinterpret_cast<const char*>(&zero32), 4);
          idxFile.write(reinterpret_cast<const char*>(&zero64), 8);
          idxFile.write(reinterpret_cast<const char*>(&zero32), 4);
        }
      }
    }
  }

  // Write meta.lcc
  writeMetaLcc(outputDir / "meta.lcc", ranges, splatsPerLod, totalSplats, numLods, config);
}

}  // namespace splat
