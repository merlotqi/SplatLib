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

#include <splat/voxel/collision-glb.h>

#include <cmath>
#include <cstring>
#include <nlohmann/json.hpp>
#include <stdexcept>


namespace splat {

namespace {

void store_le32(uint8_t* dst, uint32_t v) {
  dst[0] = static_cast<uint8_t>(v & 0xFFu);
  dst[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
  dst[2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
  dst[3] = static_cast<uint8_t>((v >> 24) & 0xFFu);
}

}  // namespace

std::vector<uint8_t> buildCollisionGlb(absl::Span<const float> positions, absl::Span<const uint32_t> indices) {
  if (positions.size() % 3 != 0) {
    throw std::invalid_argument("buildCollisionGlb: positions size must be a multiple of 3");
  }

  const size_t vertex_count = positions.size() / 3;
  const size_t index_count = indices.size();

  float min_x = 0.f, min_y = 0.f, min_z = 0.f;
  float max_x = 0.f, max_y = 0.f, max_z = 0.f;

  if (vertex_count > 0) {
    min_x = max_x = positions[0];
    min_y = max_y = positions[1];
    min_z = max_z = positions[2];
    for (size_t i = 0; i < positions.size(); i += 3) {
      const float x = positions[i];
      const float y = positions[i + 1];
      const float z = positions[i + 2];
      min_x = std::min(min_x, x);
      min_y = std::min(min_y, y);
      min_z = std::min(min_z, z);
      max_x = std::max(max_x, x);
      max_y = std::max(max_y, y);
      max_z = std::max(max_z, z);
    }
  }

  const size_t positions_byte_length = positions.size() * sizeof(float);
  const size_t indices_byte_length = indices.size() * sizeof(uint32_t);
  const size_t total_bin_size = positions_byte_length + indices_byte_length;

  nlohmann::json gltf;
  gltf["asset"] = nlohmann::json{{"version", "2.0"}, {"generator", "splat-transform"}};
  gltf["scene"] = 0;
  gltf["scenes"] = nlohmann::json::array({{{"nodes", nlohmann::json::array({0})}}});
  gltf["nodes"] = nlohmann::json::array({{{"mesh", 0}}});
  nlohmann::json mesh;
  mesh["primitives"] =
      nlohmann::json::array({nlohmann::json{{"attributes", nlohmann::json{{"POSITION", 0}}}, {"indices", 1}}});
  gltf["meshes"] = nlohmann::json::array({mesh});
  gltf["accessors"] = nlohmann::json::array(
      {nlohmann::json{{"bufferView", 0},
                      {"componentType", 5126},
                      {"count", vertex_count},
                      {"type", "VEC3"},
                      {"min", nlohmann::json::array({min_x, min_y, min_z})},
                      {"max", nlohmann::json::array({max_x, max_y, max_z})}},
       nlohmann::json{{"bufferView", 1}, {"componentType", 5125}, {"count", index_count}, {"type", "SCALAR"}}});
  gltf["bufferViews"] = nlohmann::json::array(
      {nlohmann::json{{"buffer", 0}, {"byteOffset", 0}, {"byteLength", positions_byte_length}, {"target", 34962}},
       nlohmann::json{{"buffer", 0},
                      {"byteOffset", positions_byte_length},
                      {"byteLength", indices_byte_length},
                      {"target", 34963}}});
  gltf["buffers"] = nlohmann::json::array({nlohmann::json{{"byteLength", total_bin_size}}});

  const std::string json_string = gltf.dump();
  std::vector<uint8_t> json_bytes(json_string.begin(), json_string.end());

  const size_t json_padding = (4 - (json_bytes.size() % 4)) % 4;
  const size_t json_chunk_length = json_bytes.size() + json_padding;

  const size_t bin_padding = (4 - (total_bin_size % 4)) % 4;
  const size_t bin_chunk_length = total_bin_size + bin_padding;

  const size_t total_length = 12 + 8 + json_chunk_length + 8 + bin_chunk_length;
  std::vector<uint8_t> buffer(total_length, 0);
  size_t offset = 0;

  store_le32(buffer.data() + offset, 0x46546C67u);
  offset += 4;
  store_le32(buffer.data() + offset, 2u);
  offset += 4;
  store_le32(buffer.data() + offset, static_cast<uint32_t>(total_length));
  offset += 4;

  store_le32(buffer.data() + offset, static_cast<uint32_t>(json_chunk_length));
  offset += 4;
  store_le32(buffer.data() + offset, 0x4E4F534Au);
  offset += 4;

  std::memcpy(buffer.data() + offset, json_bytes.data(), json_bytes.size());
  offset += json_bytes.size();
  for (size_t i = 0; i < json_padding; ++i) {
    buffer[offset++] = 0x20;
  }

  store_le32(buffer.data() + offset, static_cast<uint32_t>(bin_chunk_length));
  offset += 4;
  store_le32(buffer.data() + offset, 0x004E4942u);
  offset += 4;

  if (positions_byte_length > 0) {
    std::memcpy(buffer.data() + offset, positions.data(), positions_byte_length);
    offset += positions_byte_length;
  }
  if (indices_byte_length > 0) {
    std::memcpy(buffer.data() + offset, indices.data(), indices_byte_length);
    offset += indices_byte_length;
  }

  return buffer;
}

}  // namespace splat
