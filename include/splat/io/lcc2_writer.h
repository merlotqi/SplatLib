/**
 * @file splat/io/lcc2_writer.h
 * @brief Write LCC2 scenes from splat data.
 */
#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace splat {

class SplatCloud;

struct Lcc2WriteConfig {
  float cellSizeX = 30.0f;
  float cellSizeY = 30.0f;
  std::string outputFormat = "spz";  // "spz" (ply/sog via existing writers)
  std::string fileType = "Quality";  // "Quality" | "Portable"
  std::string name = "XGrids Splats";
  std::string description;
  std::string guid;  // auto-generated if empty
};

/// Write LCC2 scene from per-LOD DataTables.
/// lods[0] = LOD0 (full detail), lods[1] = LOD1, etc.
void writeLcc2(const std::filesystem::path& outputDir, const std::vector<const SplatCloud*>& lods,
               const Lcc2WriteConfig& config = {});

}  // namespace splat
