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

#include <splat/models/sog.h>

#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>

namespace splat {

Meta Meta::parseFromJson(const std::vector<uint8_t>& json) {
  std::string jsonStr(reinterpret_cast<const char*>(json.data()), json.size());

  try {
    auto j = nlohmann::json::parse(jsonStr);
    Meta meta;
    meta.version = j["version"].get<int>();
    meta.count = j["count"].get<int>();

    meta.means.mins = j["means"]["mins"].get<std::vector<float>>();
    meta.means.maxs = j["means"]["maxs"].get<std::vector<float>>();
    meta.means.files = j["means"]["files"].get<std::vector<std::string>>();

    meta.scales.codebook = j["scales"]["codebook"].get<std::vector<float>>();
    meta.scales.files = j["scales"]["files"].get<std::vector<std::string>>();

    meta.quats.files = j["quats"]["files"].get<std::vector<std::string>>();

    meta.sh0.codebook = j["sh0"]["codebook"].get<std::vector<float>>();
    meta.sh0.files = j["sh0"]["files"].get<std::vector<std::string>>();

    if (j.contains("shN") && !j["shN"].is_null()) {
      meta.shN = SHN{};
      meta.shN->count = j["shN"]["count"].get<int>();
      meta.shN->bands = j["shN"]["bands"].get<int>();
      meta.shN->codebook = j["shN"]["codebook"].get<std::vector<float>>();
      meta.shN->files = j["shN"]["files"].get<std::vector<std::string>>();
    }

    return meta;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
  }
  return {};
}

std::string Meta::encodeToJson() const {
  nlohmann::json j;
  j["version"] = version;
  j["count"] = count;

  j["means"]["mins"] = means.mins;
  j["means"]["maxs"] = means.maxs;
  j["means"]["files"] = means.files;

  j["scales"]["codebook"] = scales.codebook;
  j["scales"]["files"] = scales.files;

  j["quats"]["files"] = quats.files;

  j["sh0"]["codebook"] = sh0.codebook;
  j["sh0"]["files"] = sh0.files;

  if (shN.has_value()) {
    j["shN"]["count"] = shN->count;
    j["shN"]["bands"] = shN->bands;
    j["shN"]["codebook"] = shN->codebook;
    j["shN"]["files"] = shN->files;
  }

  return j.dump();
}

}  // namespace splat
