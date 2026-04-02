/***********************************************************************************
 *
 * splat - A C++ library for reading and writing 3D Gaussian Splatting (splat) files.
 *
 * This file is part of splat.
 *
 * splat is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 ***********************************************************************************/

#include <splat/io/glb_writer.h>
#include <splat/maths/maths.h>
#include <splat/models/data-table.h>
#include <splat/splat_version.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>


namespace splat {

namespace {

constexpr float kShC0 = 0.2820947917738781f;
constexpr uint32_t kFloatComponent = 5126;
constexpr uint32_t kUbyteComponent = 5121;
constexpr uint32_t kArrayBuffer = 34962;
constexpr uint32_t kGlbMagic = 0x46546C67u;
constexpr uint32_t kGlbVersion = 2u;
constexpr uint32_t kJsonChunkType = 0x4E4F534Au;
constexpr uint32_t kBinChunkType = 0x004E4942u;

static size_t align4(size_t n) { return (n + 3u) & ~size_t{3}; }

static void store_le32(uint8_t* dst, uint32_t v) {
  dst[0] = static_cast<uint8_t>(v & 0xFFu);
  dst[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
  dst[2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
  dst[3] = static_cast<uint8_t>((v >> 24) & 0xFFu);
}

/** Same band detection as write-glb.ts getSHBands. */
static int getShBands(const DataTable& dt) {
  int idx = -1;
  for (int i = 0; i < 45; ++i) {
    if (!dt.hasColumn("f_rest_" + std::to_string(i))) {
      idx = i;
      break;
    }
  }
  if (idx == 9) {
    return 1;
  }
  if (idx == 24) {
    return 2;
  }
  if (idx == -1) {
    return 3;
  }
  return 0;
}

struct Segment {
  std::string name;
  std::vector<uint8_t> data;
  uint32_t component_type = 0;
  std::string type_str;
  size_t count = 0;
  bool normalized = false;
  bool has_bounds = false;
  std::vector<double> min_xyz;
  std::vector<double> max_xyz;
};

static const std::vector<float>& requireFloatColumn(const DataTable& dt, const std::string& name) {
  if (!dt.hasColumn(name)) {
    throw std::invalid_argument("buildGaussianSplatGlb: missing column '" + name + "'");
  }
  const Column& col = dt.getColumnByName(name);
  if (col.getType() != ColumnType::FLOAT32) {
    throw std::invalid_argument("buildGaussianSplatGlb: column '" + name + "' must be FLOAT32");
  }
  return col.asVector<float>();
}

static void append_f32_segment(std::vector<Segment>& segments, std::string name, const std::vector<float>& floats,
                               uint32_t component_type, std::string type_str, size_t count, bool normalized = false,
                               bool with_bounds = false, float min_x = 0.f, float min_y = 0.f, float min_z = 0.f,
                               float max_x = 0.f, float max_y = 0.f, float max_z = 0.f) {
  Segment s;
  s.name = std::move(name);
  s.data.resize(floats.size() * sizeof(float));
  if (!floats.empty()) {
    std::memcpy(s.data.data(), floats.data(), s.data.size());
  }
  s.component_type = component_type;
  s.type_str = std::move(type_str);
  s.count = count;
  s.normalized = normalized;
  if (with_bounds) {
    s.has_bounds = true;
    s.min_xyz = {static_cast<double>(min_x), static_cast<double>(min_y), static_cast<double>(min_z)};
    s.max_xyz = {static_cast<double>(max_x), static_cast<double>(max_y), static_cast<double>(max_z)};
  }
  segments.push_back(std::move(s));
}

static void append_u8_segment(std::vector<Segment>& segments, std::string name, const std::vector<uint8_t>& bytes,
                              size_t count) {
  Segment s;
  s.name = std::move(name);
  s.data = bytes;
  s.component_type = kUbyteComponent;
  s.type_str = "VEC4";
  s.count = count;
  s.normalized = true;
  segments.push_back(std::move(s));
}

static std::vector<Segment> buildSegments(const DataTable& dt, size_t num_splats, int sh_bands) {
  const auto& x = requireFloatColumn(dt, "x");
  const auto& y = requireFloatColumn(dt, "y");
  const auto& z = requireFloatColumn(dt, "z");
  const auto& rot0 = requireFloatColumn(dt, "rot_0");
  const auto& rot1 = requireFloatColumn(dt, "rot_1");
  const auto& rot2 = requireFloatColumn(dt, "rot_2");
  const auto& rot3 = requireFloatColumn(dt, "rot_3");
  const auto& scale0 = requireFloatColumn(dt, "scale_0");
  const auto& scale1 = requireFloatColumn(dt, "scale_1");
  const auto& scale2 = requireFloatColumn(dt, "scale_2");
  const auto& opacity = requireFloatColumn(dt, "opacity");
  const auto& fdc0 = requireFloatColumn(dt, "f_dc_0");
  const auto& fdc1 = requireFloatColumn(dt, "f_dc_1");
  const auto& fdc2 = requireFloatColumn(dt, "f_dc_2");

  static constexpr size_t kCoeffsPerChannel[] = {0, 3, 8, 15};
  static constexpr int kShCoefCount[] = {0, 3, 5, 7};

  const int sb = std::max(0, std::min(3, sh_bands));
  const size_t coeffs_per_channel = kCoeffsPerChannel[static_cast<size_t>(sb)];

  auto check_len = [&](const std::vector<float>& v, const char* col) {
    if (v.size() != num_splats) {
      throw std::invalid_argument(std::string("buildGaussianSplatGlb: column '") + col + "' length mismatch");
    }
  };
  check_len(x, "x");
  check_len(y, "y");
  check_len(z, "z");
  check_len(rot0, "rot_0");
  check_len(rot1, "rot_1");
  check_len(rot2, "rot_2");
  check_len(rot3, "rot_3");
  check_len(scale0, "scale_0");
  check_len(scale1, "scale_1");
  check_len(scale2, "scale_2");
  check_len(opacity, "opacity");
  check_len(fdc0, "f_dc_0");
  check_len(fdc1, "f_dc_1");
  check_len(fdc2, "f_dc_2");
  const int sh_degrees = sh_bands;

  std::vector<Segment> segments;

  // POSITION
  std::vector<float> pos_data(num_splats * 3);
  float min_x = 0.f, min_y = 0.f, min_z = 0.f, max_x = 0.f, max_y = 0.f, max_z = 0.f;
  if (num_splats > 0) {
    min_x = max_x = x[0];
    min_y = max_y = y[0];
    min_z = max_z = z[0];
    for (size_t i = 0; i < num_splats; ++i) {
      pos_data[i * 3] = x[i];
      pos_data[i * 3 + 1] = y[i];
      pos_data[i * 3 + 2] = z[i];
      min_x = std::min(min_x, x[i]);
      max_x = std::max(max_x, x[i]);
      min_y = std::min(min_y, y[i]);
      max_y = std::max(max_y, y[i]);
      min_z = std::min(min_z, z[i]);
      max_z = std::max(max_z, z[i]);
    }
  }
  append_f32_segment(segments, "POSITION", pos_data, kFloatComponent, "VEC3", num_splats, false, true, min_x, min_y,
                     min_z, max_x, max_y, max_z);

  // COLOR_0
  std::vector<uint8_t> color_data(num_splats * 4);
  for (size_t i = 0; i < num_splats; ++i) {
    const int r = static_cast<int>(std::lround(std::max(0.f, std::min(255.f, (fdc0[i] * kShC0 + 0.5f) * 255.f))));
    const int g = static_cast<int>(std::lround(std::max(0.f, std::min(255.f, (fdc1[i] * kShC0 + 0.5f) * 255.f))));
    const int b = static_cast<int>(std::lround(std::max(0.f, std::min(255.f, (fdc2[i] * kShC0 + 0.5f) * 255.f))));
    const int a = static_cast<int>(std::lround(std::max(0.f, std::min(255.f, sigmoid(opacity[i]) * 255.f))));
    color_data[i * 4] = static_cast<uint8_t>(r);
    color_data[i * 4 + 1] = static_cast<uint8_t>(g);
    color_data[i * 4 + 2] = static_cast<uint8_t>(b);
    color_data[i * 4 + 3] = static_cast<uint8_t>(a);
  }
  append_u8_segment(segments, "COLOR_0", color_data, num_splats);

  // ROTATION xyzw
  std::vector<float> rot_data(num_splats * 4);
  for (size_t i = 0; i < num_splats; ++i) {
    rot_data[i * 4] = rot1[i];
    rot_data[i * 4 + 1] = rot2[i];
    rot_data[i * 4 + 2] = rot3[i];
    rot_data[i * 4 + 3] = rot0[i];
  }
  append_f32_segment(segments, "KHR_gaussian_splatting:ROTATION", rot_data, kFloatComponent, "VEC4", num_splats);

  // SCALE
  std::vector<float> scale_data(num_splats * 3);
  for (size_t i = 0; i < num_splats; ++i) {
    scale_data[i * 3] = scale0[i];
    scale_data[i * 3 + 1] = scale1[i];
    scale_data[i * 3 + 2] = scale2[i];
  }
  append_f32_segment(segments, "KHR_gaussian_splatting:SCALE", scale_data, kFloatComponent, "VEC3", num_splats);

  // OPACITY
  std::vector<float> opacity_data(num_splats);
  for (size_t i = 0; i < num_splats; ++i) {
    opacity_data[i] = sigmoid(opacity[i]);
  }
  append_f32_segment(segments, "KHR_gaussian_splatting:OPACITY", opacity_data, kFloatComponent, "SCALAR", num_splats);

  // SH degree 0
  std::vector<float> sh_dc(num_splats * 3);
  for (size_t i = 0; i < num_splats; ++i) {
    sh_dc[i * 3] = fdc0[i];
    sh_dc[i * 3 + 1] = fdc1[i];
    sh_dc[i * 3 + 2] = fdc2[i];
  }
  append_f32_segment(segments, "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0", sh_dc, kFloatComponent, "VEC3", num_splats);

  if (sh_degrees > 0) {
    std::vector<const std::vector<float>*> rest_ptrs;
    rest_ptrs.reserve(coeffs_per_channel * 3);
    for (size_t k = 0; k < coeffs_per_channel; ++k) {
      rest_ptrs.push_back(&requireFloatColumn(dt, "f_rest_" + std::to_string(k)));
      rest_ptrs.push_back(&requireFloatColumn(dt, "f_rest_" + std::to_string(k + coeffs_per_channel)));
      rest_ptrs.push_back(&requireFloatColumn(dt, "f_rest_" + std::to_string(k + 2 * coeffs_per_channel)));
    }

    int coef_offset = 0;
    for (int degree = 1; degree <= sh_degrees; ++degree) {
      const int num_coefs = kShCoefCount[degree];
      for (int c = 0; c < num_coefs; ++c) {
        const int k = coef_offset + c;
        const std::vector<float>& r_ch = *rest_ptrs[static_cast<size_t>(k) * 3];
        const std::vector<float>& g_ch = *rest_ptrs[static_cast<size_t>(k) * 3 + 1];
        const std::vector<float>& b_ch = *rest_ptrs[static_cast<size_t>(k) * 3 + 2];
        std::vector<float> buf(num_splats * 3);
        for (size_t i = 0; i < num_splats; ++i) {
          buf[i * 3] = r_ch[i];
          buf[i * 3 + 1] = g_ch[i];
          buf[i * 3 + 2] = b_ch[i];
        }
        append_f32_segment(segments,
                           "KHR_gaussian_splatting:SH_DEGREE_" + std::to_string(degree) + "_COEF_" + std::to_string(c),
                           buf, kFloatComponent, "VEC3", num_splats);
      }
      coef_offset += num_coefs;
    }
  }

  return segments;
}

}  // namespace

std::vector<uint8_t> buildGaussianSplatGlb(const DataTable& data_table) {
  const size_t num_splats = data_table.getNumRows();
  const int sh_bands = getShBands(data_table);
  std::vector<Segment> segments = buildSegments(data_table, num_splats, sh_bands);

  std::vector<size_t> offsets;
  offsets.reserve(segments.size());
  size_t total_bin = 0;
  for (const Segment& seg : segments) {
    offsets.push_back(total_bin);
    total_bin += align4(seg.data.size());
  }

  std::vector<uint8_t> bin_buffer(total_bin, 0);
  for (size_t i = 0; i < segments.size(); ++i) {
    std::memcpy(bin_buffer.data() + offsets[i], segments[i].data.data(), segments[i].data.size());
  }

  nlohmann::json buffer_views = nlohmann::json::array();
  nlohmann::json accessors = nlohmann::json::array();
  nlohmann::json attributes = nlohmann::json::object();

  for (size_t i = 0; i < segments.size(); ++i) {
    const Segment& seg = segments[i];
    buffer_views.push_back(
        {{"buffer", 0}, {"byteOffset", offsets[i]}, {"byteLength", seg.data.size()}, {"target", kArrayBuffer}});

    nlohmann::json acc = {{"bufferView", i},
                          {"byteOffset", 0},
                          {"componentType", seg.component_type},
                          {"count", seg.count},
                          {"type", seg.type_str}};
    if (seg.normalized) {
      acc["normalized"] = true;
    }
    if (seg.has_bounds) {
      acc["min"] = seg.min_xyz;
      acc["max"] = seg.max_xyz;
    }
    accessors.push_back(acc);
    attributes[seg.name] = i;
  }

  const std::string generator = std::string("splat-transform ") + splat::version;

  nlohmann::json prim = {{"attributes", attributes},
                         {"mode", 0},
                         {"extensions",
                          {{"KHR_gaussian_splatting",
                            {{"kernel", "ellipse"},
                             {"colorSpace", "srgb_rec709_display"},
                             {"sortingMethod", "cameraDistance"},
                             {"projection", "perspective"}}}}}};
  nlohmann::json mesh_obj = {{"primitives", nlohmann::json::array({prim})}};

  nlohmann::json gltf = {{"asset", {{"version", "2.0"}, {"generator", generator}}},
                         {"extensionsUsed", nlohmann::json::array({"KHR_gaussian_splatting"})},
                         {"scene", 0},
                         {"scenes", nlohmann::json::array({{{"nodes", nlohmann::json::array({0})}}})},
                         {"nodes", nlohmann::json::array({{{"mesh", 0}}})},
                         {"buffers", nlohmann::json::array({{{"byteLength", bin_buffer.size()}}})},
                         {"bufferViews", buffer_views},
                         {"accessors", accessors},
                         {"meshes", nlohmann::json::array({mesh_obj})}};

  const std::string json_string = gltf.dump();
  std::vector<uint8_t> json_bytes(json_string.begin(), json_string.end());

  const size_t json_padded_length = align4(json_bytes.size());
  const size_t bin_padded_length = align4(bin_buffer.size());
  const size_t total_length = 12 + 8 + json_padded_length + 8 + bin_padded_length;

  if (total_length > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error("buildGaussianSplatGlb: GLB too large");
  }

  std::vector<uint8_t> buffer(total_length, 0);
  size_t offset = 0;
  store_le32(buffer.data() + offset, kGlbMagic);
  offset += 4;
  store_le32(buffer.data() + offset, kGlbVersion);
  offset += 4;
  store_le32(buffer.data() + offset, static_cast<uint32_t>(total_length));
  offset += 4;

  store_le32(buffer.data() + offset, static_cast<uint32_t>(json_padded_length));
  offset += 4;
  store_le32(buffer.data() + offset, kJsonChunkType);
  offset += 4;

  std::memcpy(buffer.data() + offset, json_bytes.data(), json_bytes.size());
  offset += json_bytes.size();
  while (offset % 4 != 0) {
    buffer[offset++] = 0x20;
  }

  store_le32(buffer.data() + offset, static_cast<uint32_t>(bin_padded_length));
  offset += 4;
  store_le32(buffer.data() + offset, kBinChunkType);
  offset += 4;

  std::memcpy(buffer.data() + offset, bin_buffer.data(), bin_buffer.size());
  return buffer;
}

void writeGlb(const std::filesystem::path& filename, const DataTable* data_table) {
  if (!data_table) {
    throw std::invalid_argument("writeGlb: data_table is required");
  }
  std::vector<uint8_t> bytes = buildGaussianSplatGlb(*data_table);
  std::ofstream f(filename, std::ios::binary);
  if (!f) {
    throw std::runtime_error("writeGlb: cannot open for write: " + filename.string());
  }
  f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace splat
