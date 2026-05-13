#include <splat/io/voxel_reader.h>
#include <splat/models/data-table.h>
#include <splat/voxel/sparse-octree.h>

#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_set>
#include <vector>


namespace splat {

namespace {

constexpr double kC0 = 0.28209479177387814;

static std::filesystem::path voxelBinPath(const std::filesystem::path& json_path) {
  std::string stem = json_path.filename().string();
  static const char kSuffix[] = ".voxel.json";
  constexpr size_t kLen = sizeof(kSuffix) - 1;
  std::string lower;
  lower.reserve(stem.size());
  for (char c : stem) {
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  if (lower.size() < kLen || lower.compare(lower.size() - kLen, kLen, kSuffix) != 0) {
    throw std::invalid_argument("readVoxel: path must end with .voxel.json");
  }
  stem.replace(stem.size() - kLen, kLen, ".voxel.bin");
  return json_path.parent_path() / stem;
}

static size_t expandSolid(uint32_t morton, int depth, int tree_depth, std::vector<uint32_t>& out_morton,
                          std::vector<uint8_t>& out_solid, size_t out_count, std::vector<uint32_t>& stack_morton,
                          std::vector<int>& stack_depth) {
  stack_morton.clear();
  stack_depth.clear();
  stack_morton.push_back(morton);
  stack_depth.push_back(depth);
  while (!stack_morton.empty()) {
    int d = stack_depth.back();
    stack_depth.pop_back();
    uint32_t m = stack_morton.back();
    stack_morton.pop_back();
    if (d == tree_depth) {
      out_morton.push_back(m);
      if (out_count >= out_solid.size()) {
        out_solid.resize(std::max(out_solid.size() * 2, out_count + 1));
      }
      out_solid[out_count++] = 1;
    } else {
      for (int octant = 7; octant >= 0; --octant) {
        stack_morton.push_back(m * 8u + static_cast<uint32_t>(octant));
        stack_depth.push_back(d + 1);
      }
    }
  }
  return out_count;
}

struct LeafArrays {
  std::vector<uint32_t> morton;
  std::vector<uint8_t> is_solid;
  size_t count = 0;
};

static LeafArrays collectLeafBlocks(const std::vector<uint32_t>& nodes, int tree_depth) {
  std::unordered_set<uint32_t> is_child;
  is_child.reserve(nodes.size() * 2);
  for (size_t i = 0; i < nodes.size(); ++i) {
    uint32_t node = nodes[i];
    if (node == SOLID_LEAF_MARKER) {
      continue;
    }
    uint32_t high_byte = (node >> 24) & 0xFFu;
    if (high_byte != 0) {
      uint8_t child_mask = static_cast<uint8_t>(high_byte);
      uint32_t base_offset = node & 0x00FFFFFFu;
      for (int octant = 0; octant < 8; ++octant) {
        if (child_mask & (1 << octant)) {
          int offset = getChildOffset(child_mask, octant);
          is_child.insert(base_offset + static_cast<uint32_t>(offset));
        }
      }
    }
  }

  std::vector<uint32_t> q_node_idx;
  std::vector<uint32_t> q_morton;
  std::vector<int> q_depth;
  q_node_idx.reserve(nodes.size());
  q_morton.reserve(nodes.size());
  q_depth.reserve(nodes.size());

  uint32_t root_morton = 0;
  for (size_t i = 0; i < nodes.size(); ++i) {
    if (is_child.find(static_cast<uint32_t>(i)) == is_child.end()) {
      q_node_idx.push_back(static_cast<uint32_t>(i));
      q_morton.push_back(root_morton);
      q_depth.push_back(0);
      root_morton++;
    }
  }

  LeafArrays out;
  out.morton.reserve(nodes.size());
  size_t leaf_solid_cap = nodes.size();
  out.is_solid.resize(leaf_solid_cap);
  size_t leaf_count = 0;

  std::vector<uint32_t> expand_stack_morton;
  std::vector<int> expand_stack_depth;

  auto ensure_solid_capacity = [&](size_t needed) {
    if (needed > leaf_solid_cap) {
      leaf_solid_cap = std::max(leaf_solid_cap * 2, needed);
      out.is_solid.resize(leaf_solid_cap);
    }
  };

  size_t head = 0;
  while (head < q_node_idx.size()) {
    uint32_t node_idx = q_node_idx[head];
    uint32_t morton = q_morton[head];
    int depth = q_depth[head];
    ++head;

    uint32_t node = nodes[node_idx];

    if (node == SOLID_LEAF_MARKER) {
      int levels_to_expand = tree_depth - depth;
      if (levels_to_expand == 0) {
        out.morton.push_back(morton);
        ensure_solid_capacity(leaf_count + 1);
        out.is_solid[leaf_count++] = 1;
      } else {
        uint32_t expansion_size = 1u;
        for (int l = 0; l < levels_to_expand; ++l) {
          expansion_size *= 8u;
        }
        ensure_solid_capacity(leaf_count + expansion_size);
        leaf_count = expandSolid(morton, depth, tree_depth, out.morton, out.is_solid, leaf_count, expand_stack_morton,
                                 expand_stack_depth);
      }
    } else {
      uint32_t high_byte = (node >> 24) & 0xFFu;
      if (high_byte == 0) {
        out.morton.push_back(morton);
        ensure_solid_capacity(leaf_count + 1);
        out.is_solid[leaf_count++] = 0;
      } else {
        uint8_t child_mask = static_cast<uint8_t>(high_byte);
        uint32_t base_offset = node & 0x00FFFFFFu;
        for (int octant = 0; octant < 8; ++octant) {
          if (child_mask & (1 << octant)) {
            int offset = getChildOffset(child_mask, octant);
            q_node_idx.push_back(base_offset + static_cast<uint32_t>(offset));
            q_morton.push_back(morton * 8u + static_cast<uint32_t>(octant));
            q_depth.push_back(depth + 1);
          }
        }
      }
    }
  }

  out.is_solid.resize(leaf_count);
  out.count = leaf_count;
  return out;
}

static std::unique_ptr<DataTable> emptyVoxelTable() {
  std::vector<Column> cols;
  const char* names[] = {"x",     "y",     "z",     "scale_0", "scale_1", "scale_2", "rot_0",
                         "rot_1", "rot_2", "rot_3", "f_dc_0",  "f_dc_1",  "f_dc_2",  "opacity"};
  for (const char* n : names) {
    cols.push_back({n, std::vector<float>{}});
  }
  return std::make_unique<DataTable>(std::move(cols));
}

}  // namespace

std::unique_ptr<DataTable> readVoxel(const std::filesystem::path& voxel_json_path) {
  std::ifstream json_file(voxel_json_path);
  if (!json_file) {
    throw std::runtime_error("readVoxel: cannot open JSON: " + voxel_json_path.string());
  }
  nlohmann::json j;
  json_file >> j;

  const std::string version = j.at("version").get<std::string>();
  if (version != "1.0" && version != "1.1") {
    throw std::runtime_error("Unsupported voxel format version: " + version);
  }

  const auto& grid_min = j.at("gridBounds").at("min");
  float gx0 = grid_min.at(0).get<float>();
  float gy0 = grid_min.at(1).get<float>();
  float gz0 = grid_min.at(2).get<float>();

  float voxel_resolution = j.at("voxelResolution").get<float>();
  int tree_depth = j.at("treeDepth").get<int>();
  uint32_t node_count = j.at("nodeCount").get<uint32_t>();
  uint32_t leaf_data_count = j.at("leafDataCount").get<uint32_t>();

  const std::filesystem::path bin_path = voxelBinPath(voxel_json_path);
  std::ifstream bin_file(bin_path, std::ios::binary);
  if (!bin_file) {
    throw std::runtime_error("Failed to load voxel binary file '" + bin_path.filename().string() +
                             "'. Ensure it exists alongside " + voxel_json_path.filename().string() + ".");
  }

  const size_t expected_size = static_cast<size_t>(node_count + leaf_data_count) * 4u;
  bin_file.seekg(0, std::ios::end);
  const std::streampos end_pos = bin_file.tellg();
  bin_file.seekg(0, std::ios::beg);
  const size_t file_size = end_pos == std::streampos(-1) ? 0u : static_cast<size_t>(end_pos);
  if (file_size < expected_size) {
    throw std::runtime_error("Voxel binary file truncated: expected " + std::to_string(expected_size) + " bytes, got " +
                             std::to_string(file_size));
  }

  std::vector<uint32_t> nodes(node_count);
  if (node_count > 0) {
    bin_file.read(reinterpret_cast<char*>(nodes.data()), static_cast<std::streamsize>(node_count * sizeof(uint32_t)));
  }
  std::vector<uint32_t> leaf_data(leaf_data_count);
  if (leaf_data_count > 0) {
    bin_file.read(reinterpret_cast<char*>(leaf_data.data()),
                  static_cast<std::streamsize>(leaf_data_count * sizeof(uint32_t)));
  }
  (void)leaf_data;

  LeafArrays leaves = collectLeafBlocks(nodes, tree_depth);
  if (leaves.count == 0) {
    return emptyVoxelTable();
  }

  const float block_size = 4.f * voxel_resolution;
  const float splat_scale = std::log(block_size * 0.4f);
  const size_t n = leaves.count;

  const float solid_r = static_cast<float>((0.9 - 0.5) / kC0);
  const float solid_g = static_cast<float>((0.1 - 0.5) / kC0);
  const float solid_b = static_cast<float>((0.1 - 0.5) / kC0);

  std::vector<float> x_arr(n), y_arr(n), z_arr(n);
  std::vector<float> s0(n), s1(n), s2(n);
  std::vector<float> r0(n), r1(n), r2(n), r3(n);
  std::vector<float> f0(n), f1(n), f2(n);
  std::vector<float> op(n);

  for (size_t i = 0; i < n; ++i) {
    uint32_t morton = leaves.morton[i];
    uint32_t bx = 0, by = 0, bz = 0;
    uint32_t bit = 1;
    uint32_t m = morton;
    while (m > 0) {
      uint32_t triplet = m % 8u;
      if (triplet & 1u) bx |= bit;
      if (triplet & 2u) by |= bit;
      if (triplet & 4u) bz |= bit;
      bit <<= 1;
      m /= 8u;
    }

    x_arr[i] = gx0 + (static_cast<float>(bx) + 0.5f) * block_size;
    y_arr[i] = gy0 + (static_cast<float>(by) + 0.5f) * block_size;
    z_arr[i] = gz0 + (static_cast<float>(bz) + 0.5f) * block_size;

    s0[i] = splat_scale;
    s1[i] = splat_scale;
    s2[i] = splat_scale;

    r0[i] = 1.f;
    r1[i] = 0.f;
    r2[i] = 0.f;
    r3[i] = 0.f;

    if (leaves.is_solid[i]) {
      f0[i] = solid_r;
      f1[i] = solid_g;
      f2[i] = solid_b;
    } else {
      double gray = 0.3 + std::fmod(static_cast<double>(morton) * 0.618033988749895, 1.0);
      float g = static_cast<float>(gray);
      float sh = (g - 0.5f) / static_cast<float>(kC0);
      f0[i] = sh;
      f1[i] = sh;
      f2[i] = sh;
    }
    op[i] = 5.f;
  }

  std::vector<Column> cols;
  cols.push_back({"x", std::move(x_arr)});
  cols.push_back({"y", std::move(y_arr)});
  cols.push_back({"z", std::move(z_arr)});
  cols.push_back({"scale_0", std::move(s0)});
  cols.push_back({"scale_1", std::move(s1)});
  cols.push_back({"scale_2", std::move(s2)});
  cols.push_back({"rot_0", std::move(r0)});
  cols.push_back({"rot_1", std::move(r1)});
  cols.push_back({"rot_2", std::move(r2)});
  cols.push_back({"rot_3", std::move(r3)});
  cols.push_back({"f_dc_0", std::move(f0)});
  cols.push_back({"f_dc_1", std::move(f1)});
  cols.push_back({"f_dc_2", std::move(f2)});
  cols.push_back({"opacity", std::move(op)});

  return std::make_unique<DataTable>(std::move(cols));
}

}  // namespace splat
