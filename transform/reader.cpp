#include <splat/splat.h>
#include <absl/strings/match.h>

#include <filesystem>

#include "process.h"
#include "options.h"

using namespace splat;

static std::string getInputFormat(std::string filename) {
  if (absl::EndsWithIgnoreCase(filename, ".ksplat")) {
    return "ksplat";
  } else if (absl::EndsWithIgnoreCase(filename, ".splat")) {
    return "splat";
  } else if (absl::EndsWithIgnoreCase(filename, ".sog") || absl::EndsWithIgnoreCase(filename, "meta.json")) {
    return "sog";
  } else if (absl::EndsWithIgnoreCase(filename, ".ply")) {
    return "ply";
  } else if (absl::EndsWithIgnoreCase(filename, ".spz")) {
    return "spz";
  } else if (absl::EndsWithIgnoreCase(filename, ".lcc")) {
    return "lcc";
  } else if (absl::EndsWithIgnoreCase(filename, ".voxel.json")) {
    return "voxel";
  }
  throw std::runtime_error("Unsupported input file type" + filename);
}

std::vector<std::unique_ptr<DataTable>> readFile(const std::string& filename, const Options& options,
                                                        const std::vector<Param>& params) {
  const auto inputFormat = getInputFormat(filename);
  std::vector<std::unique_ptr<DataTable>> results;

  LOG_INFO("reading %s...", filename.c_str());

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
    results.emplace_back(readVoxel(std::filesystem::path(filename)));
  }

  return results;
}
