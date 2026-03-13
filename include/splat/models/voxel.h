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

struct VoxelMetadata {
  /** File format version */
  std::string version;

  /** Grid bounds aligned to 4x4x4 block boundaries */
  struct {
    std::vector<double> min;
    std::vector<double> max;
  } gridBounds;

  /** Original Gaussian scene bounds */
  struct {
    std::vector<double> min;
    std::vector<double> max;
  } sceneBounds;

  /** Size of each voxel in world units */
  double voxelResolution{0.0};

  /** Voxels per leaf dimension (always 4) */
  int leafSize{4};

  /** Maximum tree depth */
  int treeDepth{0};

  /** Number of interior nodes */
  int numInteriorNodes{0};

  /** Number of mixed leaf nodes */
  int numMixedLeaves{0};

  /** Total number of Uint32 entries in the nodes array */
  int nodeCount{0};

  /** Total number of Uint32 entries in the leafData array */
  int leafDataCount{0};

  VoxelMetadata(const std::string& json);

  std::string dump() const;
};

}  // namespace splat
