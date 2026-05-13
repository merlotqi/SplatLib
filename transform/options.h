#pragma once

#include <filesystem>
#include <vector>

namespace splat {

/**
 * @brief Corresponds to TypeScript type Options.
 * Contains all configuration options for the application.
 */
struct Options {
  // Basic Options
  bool overwrite;
  bool help;
  bool version;
  bool quiet;
  int iterations;
  bool listGpus;

  // Device selection: -1 = auto, -2 = CPU, 0+ = GPU index
  int device;

  // lcc input options
  std::vector<int> lodSelect;

  // html output options
  std::filesystem::path viewerSettingsPath;
  bool unbundled;

  // lod output options
  int lodChunkCount;
  int lodChunkExtent;
  bool lodBundle;

  /**
   * @brief Constructor for Options.
   * Initializes all members with default values.
   */
  Options() {
    // Basic Options defaults
    overwrite = false;
    help = false;
    version = false;
    quiet = false;
    iterations = 1;
    listGpus = false;
    device = -1;  // -1 = auto

    // lcc input options defaults
    // lodSelect is an empty vector by default (no initialization needed here,
    // as std::vector is default-constructed to empty)

    // html output options defaults
    viewerSettingsPath.clear();
    unbundled = false;

    // lod output options defaults
    lodChunkCount = 64;
    lodChunkExtent = 16;
    lodBundle = true;
  }
};

}  // namespace splat
