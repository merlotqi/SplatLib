#include <absl/strings/match.h>
#include <splat/splat.h>

#include <iostream>
#include <random>

#include "options.h"

using namespace splat;

std::string getOutputFormat(std::string filename) {
  if (absl::EndsWithIgnoreCase(filename, ".csv")) {
    return "csv";
  }
  if (absl::EndsWithIgnoreCase(filename, "lod-meta.json")) {
    return "lod";
  }
  if (absl::EndsWithIgnoreCase(filename, ".sog")) {
    return "sog-bundle";
  } else if (absl::EndsWithIgnoreCase(filename, "meta.json")) {
    return "sog";
  }
  if (absl::EndsWithIgnoreCase(filename, ".compressed.ply")) {
    return "compressed-ply";
  }
  if (absl::EndsWithIgnoreCase(filename, ".ply")) {
    return "ply";
  }

  throw std::runtime_error("Unsupported output file type: " + std::string(filename));
}

void writeFile(const std::string& filename, DataTable* dataTable, DataTable* envDataTable, const Options& options) {
  auto getRandomHex = [](size_t length) -> std::string {
    static const char* const lut = "0123456789abcdef";
    std::string res;
    res.reserve(length);
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distribution(0, 15);
    for (size_t i = 0; i < length; ++i) {
      res += lut[distribution(generator)];
    }
    return res;
  };

  std::string outputFormat = getOutputFormat(filename);

  std::cout << "writing '" << filename << "'..." << "\n";

  try {
    if (outputFormat == "csv") {
      writeCSV(filename, dataTable);
    } else if (outputFormat == "sog" || outputFormat == "sog-bundle") {
      writeSog(filename, dataTable, outputFormat == "sog-bundle", options.iterations);
    } else if (outputFormat == "lod") {
      if (!dataTable->hasColumn("lod")) {
        dataTable->addColumn({"lod", std::vector<float>(dataTable->getNumRows())});
      }
      writeLod(filename, dataTable, envDataTable, false, options.iterations, options.lodChunkCount,
               options.lodChunkExtent);
    } else if (outputFormat == "compressed-ply") {
      writeCompressedPly(filename, dataTable);
    } else if (outputFormat == "ply") {
      PlyData ply;
      ply.elements.push_back({"vertex", dataTable->clone()});
      writePly(filename, ply);
    }
  } catch (...) {
    throw;
  }
}
