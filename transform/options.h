/***********************************************************************************
 *
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
 *
 ***********************************************************************************/

#pragma once

#include <string>
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
  std::string viewerSettingsPath;
  bool unbundled;

  // lod output options
  int lodChunkCount;
  int lodChunkExtent;

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
    viewerSettingsPath = "";  // Default empty string
    unbundled = false;

    // lod output options defaults
    lodChunkCount = 64;
    lodChunkExtent = 16;
  }
};

}  // namespace splat
