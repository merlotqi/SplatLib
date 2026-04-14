#include <splat/splat.h>
#include <absl/strings/match.h>

#include <filesystem>
#include <string>

#include "process.h"
#include "options.h"

using namespace splat;

namespace {

static std::string getInputFormat(const std::filesystem::path& filename) {
  const std::string u8 = filename.u8string();
  if (absl::EndsWithIgnoreCase(u8, ".ksplat")) {
    return "ksplat";
  }
  if (absl::EndsWithIgnoreCase(u8, ".splat")) {
    return "splat";
  }
  if (absl::EndsWithIgnoreCase(u8, ".sog") || absl::EndsWithIgnoreCase(u8, "meta.json")) {
    return "sog";
  }
  if (absl::EndsWithIgnoreCase(u8, ".ply")) {
    return "ply";
  }
  if (absl::EndsWithIgnoreCase(u8, ".spz")) {
    return "spz";
  }
  if (absl::EndsWithIgnoreCase(u8, ".lcc")) {
    return "lcc";
  }
  if (absl::EndsWithIgnoreCase(u8, ".voxel.json")) {
    return "voxel";
  }
  throw std::runtime_error("Unsupported input file type" + u8);
}

}  // namespace

std::vector<std::unique_ptr<DataTable>> readFile(const std::filesystem::path& filename, const Options& options,
                                                 const std::vector<Param>& params) {
  (void)params;
  const auto inputFormat = getInputFormat(filename);
  std::vector<std::unique_ptr<DataTable>> results;

  LOG_INFO("reading %s...", filename.u8string().c_str());

  if (inputFormat == "ksplat") {
    results.emplace_back(readKsplat(filename));
  } else if (inputFormat == "splat") {
    results.emplace_back(readSplat(filename));
  } else if (inputFormat == "sog") {
    results.emplace_back(readSog(filename, filename));
  } else if (inputFormat == "ply") {
    results.emplace_back(readPly(filename));
  } else if (inputFormat == "spz") {
    results.emplace_back(readSpz(filename));
  } else if (inputFormat == "lcc") {
    results = readLcc(filename, filename, options.lodSelect);
  } else if (inputFormat == "voxel") {
    results.emplace_back(readVoxel(filename));
  }

  return results;
}
