#include <splat/models/sog.h>

#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>

namespace splat {

namespace {

std::vector<std::filesystem::path> jsonStringArrayToPaths(const nlohmann::json& arr) {
  std::vector<std::filesystem::path> out;
  for (const auto& el : arr) {
    out.push_back(std::filesystem::u8path(el.get<std::string>()));
  }
  return out;
}

nlohmann::json pathsToJsonArray(const std::vector<std::filesystem::path>& paths) {
  auto j = nlohmann::json::array();
  for (const auto& p : paths) {
    j.push_back(p.u8string());
  }
  return j;
}

}  // namespace

Meta Meta::parseFromJson(const std::vector<uint8_t>& json) {
  std::string jsonStr(reinterpret_cast<const char*>(json.data()), json.size());

  try {
    auto j = nlohmann::json::parse(jsonStr);
    Meta meta;
    meta.version = j["version"].get<int>();
    meta.count = j["count"].get<int>();

    meta.means.mins = j["means"]["mins"].get<std::vector<float>>();
    meta.means.maxs = j["means"]["maxs"].get<std::vector<float>>();
    meta.means.files = jsonStringArrayToPaths(j["means"]["files"]);

    meta.scales.codebook = j["scales"]["codebook"].get<std::vector<float>>();
    meta.scales.files = jsonStringArrayToPaths(j["scales"]["files"]);

    meta.quats.files = jsonStringArrayToPaths(j["quats"]["files"]);

    meta.sh0.codebook = j["sh0"]["codebook"].get<std::vector<float>>();
    meta.sh0.files = jsonStringArrayToPaths(j["sh0"]["files"]);

    if (j.contains("shN") && !j["shN"].is_null()) {
      meta.shN = SHN{};
      meta.shN->count = j["shN"]["count"].get<int>();
      meta.shN->bands = j["shN"]["bands"].get<int>();
      meta.shN->codebook = j["shN"]["codebook"].get<std::vector<float>>();
      meta.shN->files = jsonStringArrayToPaths(j["shN"]["files"]);
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
  j["asset"]["generator"] = asset.generator;

  j["means"]["mins"] = means.mins;
  j["means"]["maxs"] = means.maxs;
  j["means"]["files"] = pathsToJsonArray(means.files);

  j["scales"]["codebook"] = scales.codebook;
  j["scales"]["files"] = pathsToJsonArray(scales.files);

  j["quats"]["files"] = pathsToJsonArray(quats.files);

  j["sh0"]["codebook"] = sh0.codebook;
  j["sh0"]["files"] = pathsToJsonArray(sh0.files);

  if (shN.has_value()) {
    j["shN"]["count"] = shN->count;
    j["shN"]["bands"] = shN->bands;
    j["shN"]["codebook"] = shN->codebook;
    j["shN"]["files"] = pathsToJsonArray(shN->files);
  }

  return j.dump();
}

}  // namespace splat
