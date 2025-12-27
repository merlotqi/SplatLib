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

#include <splat/readers/decompress_ply.h>

namespace splat {

static constexpr size_t CHUNK_SIZE = 256;

static float lerp(float a, float b, float t) { return a * (1.0f - t) + b * t; }

static float unpackUnorm(uint32_t value, int bits) {
  uint32_t t = (1u << bits) - 1u;
  return static_cast<float>(value & t) / static_cast<float>(t);
}

static Eigen::Vector3f unpack111011(uint32_t value) {
  return Eigen::Vector3f(unpackUnorm(value >> 21, 11), unpackUnorm(value >> 11, 10), unpackUnorm(value, 11));
}

static Eigen::Vector4f unpack8888(uint32_t value) {
  return Eigen::Vector4f(unpackUnorm(value >> 24, 8), unpackUnorm(value >> 16, 8), unpackUnorm(value >> 8, 8),
                         unpackUnorm(value, 8));
}

static Eigen::Vector4f unpackRot(uint32_t value) {
  const float norm = 1.0f / (std::sqrt(2.0f) * 0.5f);
  float a = (unpackUnorm(value >> 20, 10) - 0.5f) * norm;
  float b = (unpackUnorm(value >> 10, 10) - 0.5f) * norm;
  float c = (unpackUnorm(value, 10) - 0.5f) * norm;
  float m = std::sqrt(std::max(0.0f, 1.0f - (a * a + b * b + c * c)));

  uint32_t which = value >> 30;
  switch (which) {
    case 0:
      return Eigen::Vector4f(m, a, b, c);
    case 1:
      return Eigen::Vector4f(a, m, b, c);
    case 2:
      return Eigen::Vector4f(a, b, m, c);
    default:
      return Eigen::Vector4f(a, b, c, m);
  }
}

bool isCompressedPly(const PlyData* ply) {
  auto hasShape = [](const DataTable* dataTable, const std::vector<std::string>& columns, ColumnType type) {
    for (const auto& name : columns) {
      if (dataTable->hasColumn(name)) {
        if (dataTable->getColumnByName(name).getType() != type) {
          return false;
        }
      }
    }
    return true;
  };

  static std::vector<std::string> chunkProperties = {
      "min_x",       "min_y",       "min_z",       "max_x",       "max_y",       "max_z",
      "min_scale_x", "min_scale_y", "min_scale_z", "max_scale_x", "max_scale_y", "max_scale_z",
      "min_r",       "min_g",       "min_b",       "max_r",       "max_g",       "max_b"};

  static std::vector<std::string> vertexProperties = {"packed_position", "packed_rotation", "packed_scale",
                                                      "packed_color"};

  const auto numElements = ply->elements.size();
  if (numElements != 2 && numElements != 3) return false;

  auto chunkIt =
      std::find_if(ply->elements.begin(), ply->elements.end(), [](const auto& e) { return e.name == "chunk"; });
  if (chunkIt == ply->elements.end()) return false;
  if (!hasShape(chunkIt->dataTable.get(), chunkProperties, ColumnType::FLOAT32)) return false;

  auto vertexIt =
      std::find_if(ply->elements.begin(), ply->elements.end(), [](const auto& e) { return e.name == "vertex"; });
  if (vertexIt == ply->elements.end()) return false;
  if (!hasShape(vertexIt->dataTable.get(), vertexProperties, ColumnType::UINT32)) return false;

  size_t expectedChunkRows = (vertexIt->dataTable->getNumRows() + CHUNK_SIZE - 1) / CHUNK_SIZE;
  if (expectedChunkRows != chunkIt->dataTable->getNumRows()) {
    return false;
  }

  if (numElements == 3) {
    auto shIt = std::find_if(ply->elements.begin(), ply->elements.end(), [](const auto& e) { return e.name == "sh"; });
    if (shIt == ply->elements.end()) return false;

    auto&& shData = shIt->dataTable;

    uint32_t numCols = shData->getNumColumns();
    if (numCols != 9 && numCols != 24 && numCols != 45) {
      return false;
    }

    for (uint32_t i = 0; i < numCols; ++i) {
      std::string colName = "f_rest_" + std::to_string(i);
      if (shData->hasColumn(colName)) {
        if (shData->getColumnByName(colName).getType() != ColumnType::UINT8) {
          return false;
        }
      }
    }

    if (shData->getNumRows() != vertexIt->dataTable->getNumRows()) {
      return false;
    }
  }

  return true;
}

std::unique_ptr<DataTable> decompressPly(const PlyData* ply) {
  auto chunkIt =
      std::find_if(ply->elements.begin(), ply->elements.end(), [](const auto& e) { return e.name == "chunk"; });
  if (chunkIt == ply->elements.end()) throw std::runtime_error("Missing 'chunk' element");
  const DataTable& chunkData = *chunkIt->dataTable;

  auto vertexIt =
      std::find_if(ply->elements.begin(), ply->elements.end(), [](const auto& e) { return e.name == "vertex"; });
  if (vertexIt == ply->elements.end()) throw std::runtime_error("Missing 'vertex' element");
  const DataTable& vertexData = *vertexIt->dataTable;

  auto packed_pos = vertexData.getColumnByName("packed_position").asSpan<uint32_t>();
  auto packed_rot = vertexData.getColumnByName("packed_rotation").asSpan<uint32_t>();
  auto packed_scale = vertexData.getColumnByName("packed_scale").asSpan<uint32_t>();
  auto packed_color = vertexData.getColumnByName("packed_color").asSpan<uint32_t>();

  size_t numSplats = vertexData.getNumRows();
  constexpr int CHUNK_SIZE = 256;

  auto result = std::make_unique<DataTable>();
  std::vector<std::string> targetCols = {"x",     "y",     "z",     "f_dc_0", "f_dc_1",  "f_dc_2",  "opacity",
                                         "rot_0", "rot_1", "rot_2", "rot_3",  "scale_0", "scale_1", "scale_2"};
  for (const auto& name : targetCols) {
    result->addColumn(Column{name, std::vector<float>(numSplats)});
  }

  auto ox = result->getColumnByName("x").asSpan<float>();
  auto oy = result->getColumnByName("y").asSpan<float>();
  auto oz = result->getColumnByName("z").asSpan<float>();
  auto of = std::array<absl::Span<float>, 3>{result->getColumnByName("f_dc_0").asSpan<float>(),
                                             result->getColumnByName("f_dc_1").asSpan<float>(),
                                             result->getColumnByName("f_dc_2").asSpan<float>()};
  auto oo = result->getColumnByName("opacity").asSpan<float>();
  auto orot = std::array<absl::Span<float>, 4>{
      result->getColumnByName("rot_0").asSpan<float>(), result->getColumnByName("rot_1").asSpan<float>(),
      result->getColumnByName("rot_2").asSpan<float>(), result->getColumnByName("rot_3").asSpan<float>()};
  auto os = std::array<absl::Span<float>, 3>{result->getColumnByName("scale_0").asSpan<float>(),
                                             result->getColumnByName("scale_1").asSpan<float>(),
                                             result->getColumnByName("scale_2").asSpan<float>()};

  const float SH_C0 = 0.28209479177387814f;

  for (size_t i = 0; i < numSplats; ++i) {
    size_t ci = i / CHUNK_SIZE;

    Eigen::Vector3f p_raw = unpack111011(packed_pos[i]);
    Eigen::Vector4f r_raw = unpackRot(packed_rot[i]);
    Eigen::Vector3f s_raw = unpack111011(packed_scale[i]);
    Eigen::Vector4f c_raw = unpack8888(packed_color[i]);

    ox[i] = lerp(chunkData.getColumnByName("min_x").getValue<float>(ci),
                 chunkData.getColumnByName("max_x").getValue<float>(ci), p_raw.x());
    oy[i] = lerp(chunkData.getColumnByName("min_y").getValue<float>(ci),
                 chunkData.getColumnByName("max_y").getValue<float>(ci), p_raw.y());
    oz[i] = lerp(chunkData.getColumnByName("min_z").getValue<float>(ci),
                 chunkData.getColumnByName("max_z").getValue<float>(ci), p_raw.z());

    for (int j = 0; j < 4; ++j) orot[j][i] = r_raw[j];

    os[0][i] = lerp(chunkData.getColumnByName("min_scale_x").getValue<float>(ci),
                    chunkData.getColumnByName("max_scale_x").getValue<float>(ci), s_raw.x());
    os[1][i] = lerp(chunkData.getColumnByName("min_scale_y").getValue<float>(ci),
                    chunkData.getColumnByName("max_scale_y").getValue<float>(ci), s_raw.y());
    os[2][i] = lerp(chunkData.getColumnByName("min_scale_z").getValue<float>(ci),
                    chunkData.getColumnByName("max_scale_z").getValue<float>(ci), s_raw.z());

    float cr = lerp(chunkData.getColumnByName("min_r").getValue<float>(ci),
                    chunkData.getColumnByName("max_r").getValue<float>(ci), c_raw.x());
    float cg = lerp(chunkData.getColumnByName("min_g").getValue<float>(ci),
                    chunkData.getColumnByName("max_g").getValue<float>(ci), c_raw.y());
    float cb = lerp(chunkData.getColumnByName("min_b").getValue<float>(ci),
                    chunkData.getColumnByName("max_b").getValue<float>(ci), c_raw.z());

    of[0][i] = (cr - 0.5f) / SH_C0;
    of[1][i] = (cg - 0.5f) / SH_C0;
    of[2][i] = (cb - 0.5f) / SH_C0;

    oo[i] = -std::log(1.0f / std::max(1e-7f, c_raw.w()) - 1.0f);
  }

  auto shIt = std::find_if(ply->elements.begin(), ply->elements.end(), [](const auto& e) { return e.name == "sh"; });
  if (shIt != ply->elements.end()) {
    const DataTable& shData = *shIt->dataTable;
    for (size_t k = 0; k < shData.getNumColumns(); ++k) {
      const Column& col = shData.getColumn(k);
      auto src = col.asSpan<uint8_t>();
      std::vector<float> dst(numSplats);

      for (size_t i = 0; i < numSplats; ++i) {
        float n = (src[i] == 0) ? 0.0f : (src[i] == 255) ? 1.0f : (static_cast<float>(src[i]) + 0.5f) / 256.0f;
        dst[i] = (n - 0.5f) * 8.0f;
      }
      result->addColumn(Column{col.name, std::move(dst)});
    }
  }

  return result;
}

}  // namespace splat
