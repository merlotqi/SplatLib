#include <absl/strings/match.h>
#include <splat/splat.h>

#include <filesystem>
#include <iostream>
#include <string>

#include "options.h"

using namespace splat;

std::string getOutputFormat(const std::filesystem::path& filename) {
  const std::string u8 = filename.u8string();
  if (absl::EndsWithIgnoreCase(u8, ".csv")) {
    return "csv";
  }
  if (absl::EndsWithIgnoreCase(u8, "lod-meta.json")) {
    return "lod";
  }
  if (absl::EndsWithIgnoreCase(u8, ".sog")) {
    return "sog-bundle";
  }
  if (absl::EndsWithIgnoreCase(u8, "meta.json")) {
    return "sog";
  }
  if (absl::EndsWithIgnoreCase(u8, ".compressed.ply")) {
    return "compressed-ply";
  }
  if (absl::EndsWithIgnoreCase(u8, ".ply")) {
    return "ply";
  }

  throw std::runtime_error("Unsupported output file type: " + u8);
}

void writeFile(const std::filesystem::path& filename, DataTable* dataTable, DataTable* envDataTable,
               const Options& options) {
  std::string outputFormat = getOutputFormat(filename);

  std::cout << "writing '" << filename.u8string() << "'..." << "\n";

  try {
    if (outputFormat == "csv") {
      writeCSV(filename, dataTable);
    } else if (outputFormat == "sog" || outputFormat == "sog-bundle") {
      writeSog(filename, dataTable, outputFormat == "sog-bundle", options.iterations);
    } else if (outputFormat == "lod") {
      if (!dataTable->hasColumn("lod")) {
        dataTable->addColumn({"lod", std::vector<float>(dataTable->getNumRows())});
      }
      writeLod(filename, dataTable, envDataTable, options.lodBundle, options.iterations, options.lodChunkCount,
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
