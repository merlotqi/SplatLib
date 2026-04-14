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

#include "splat/voxel/sparse-octree.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace splat {

/* -------------------------------------------------------------------------- */
/*                       BlockAccumulator Implementation                      */
/* -------------------------------------------------------------------------- */

void BlockAccumulator::addBlock(uint32_t morton, uint32_t lo, uint32_t hi) {
  if (isEmpty(lo, hi)) return;
  if (isSolid(lo, hi)) {
    solidMorton_.push_back(morton);
  } else {
    mixedMorton_.push_back(morton);
    mixedMasks_.push_back(lo);
    mixedMasks_.push_back(hi);
  }
}

BlockAccumulator::MixedBlocks BlockAccumulator::getMixedBlocks() const { return {mixedMorton_, mixedMasks_}; }

const std::vector<uint32_t>& BlockAccumulator::getSolidBlocks() const { return solidMorton_; }

size_t BlockAccumulator::count() const { return mixedMorton_.size() + solidMorton_.size(); }
size_t BlockAccumulator::mixedCount() const { return mixedMorton_.size(); }
size_t BlockAccumulator::solidCount() const { return solidMorton_.size(); }

void BlockAccumulator::clear() {
  mixedMorton_.clear();
  mixedMasks_.clear();
  solidMorton_.clear();
}

/* -------------------------------------------------------------------------- */
/*                      BlockType Enumeration (internal)                      */
/* -------------------------------------------------------------------------- */

enum class BlockType : uint8_t {
  Empty = 0,
  Solid = 1,
  Mixed = 2
};

/* -------------------------------------------------------------------------- */
/*                             Octree Construction                            */
/* -------------------------------------------------------------------------- */

Bounds alignGridBounds(float minX, float minY, float minZ, float maxX, float maxY, float maxZ, float voxelResolution) {
  const float blockSize = 4.0f * voxelResolution;
  return {Eigen::Vector3f(std::floor(minX / blockSize) * blockSize, std::floor(minY / blockSize) * blockSize,
                          std::floor(minZ / blockSize) * blockSize),
          Eigen::Vector3f(std::ceil(maxX / blockSize) * blockSize, std::ceil(maxY / blockSize) * blockSize,
                          std::ceil(maxZ / blockSize) * blockSize)};
}

SparseOctree buildSparseOctree(const BlockAccumulator& accumulator, const Bounds& gridBounds, const Bounds& sceneBounds,
                               float voxelResolution) {
  const auto mixed = accumulator.getMixedBlocks();
  const auto& solid = accumulator.getSolidBlocks();
  const size_t totalBlocks = mixed.morton.size() + solid.size();

  // Combine blocks into SoA arrays
  std::vector<uint32_t> mortons(totalBlocks);
  std::vector<uint8_t> types(totalBlocks);
  std::vector<int> maskIndices(totalBlocks);

  size_t idx = 0;
  for (size_t i = 0; i < mixed.morton.size(); i++) {
    mortons[idx] = mixed.morton[i];
    types[idx] = static_cast<uint8_t>(BlockType::Mixed);
    maskIndices[idx] = static_cast<int>(i);
    idx++;
  }
  for (size_t i = 0; i < solid.size(); i++) {
    mortons[idx] = solid[i];
    types[idx] = static_cast<uint8_t>(BlockType::Solid);
    maskIndices[idx] = -1;
    idx++;
  }

  // Co-sort by Morton code using an index permutation array
  std::vector<size_t> sortOrder(totalBlocks);
  for (size_t i = 0; i < totalBlocks; i++) sortOrder[i] = i;
  std::sort(sortOrder.begin(), sortOrder.end(), [&](size_t a, size_t b) { return mortons[a] < mortons[b]; });

  std::vector<uint32_t> sortedMortons(totalBlocks);
  std::vector<uint8_t> sortedTypes(totalBlocks);
  std::vector<int> sortedMaskIndices(totalBlocks);
  for (size_t i = 0; i < totalBlocks; i++) {
    const size_t si = sortOrder[i];
    sortedMortons[i] = mortons[si];
    sortedTypes[i] = types[si];
    sortedMaskIndices[i] = maskIndices[si];
  }

  // Calculate tree depth based on grid size
  const Eigen::Vector3f gridSize = gridBounds.max - gridBounds.min;
  const float blockSize = voxelResolution * 4.0f;
  const auto blocksPerAxis = std::max({static_cast<int>(std::ceil(gridSize.x() / blockSize)),
                                       static_cast<int>(std::ceil(gridSize.y() / blockSize)),
                                       static_cast<int>(std::ceil(gridSize.z() / blockSize))});
  const int treeDepth = std::max(1, static_cast<int>(std::ceil(std::log2(blocksPerAxis))));

  // Store level data for each tree level
  struct LevelData {
    std::vector<uint32_t> mortons;
    std::vector<uint8_t> types;
    std::vector<int> maskIndices;
    std::vector<uint8_t> childMasks;
  };

  std::vector<LevelData> levels;
  levels.reserve(static_cast<size_t>(treeDepth) + 1);

  // Current level data starts as the sorted leaf blocks
  std::vector<uint32_t>* curMortons = &sortedMortons;
  std::vector<uint8_t>* curTypes = &sortedTypes;
  std::vector<int>* curMaskIndices = &sortedMaskIndices;
  std::vector<uint8_t> curChildMasks(totalBlocks, 0);

  // Build up level by level
  int actualDepth = treeDepth;

  for (int level = 0; level < treeDepth; level++) {
    // Save current level before building the next one above
    levels.push_back({*curMortons, *curTypes, *curMaskIndices, curChildMasks});

    // Build next level using linear scan on sorted data
    const size_t n = curMortons->size();
    std::vector<uint32_t> nextMortons;
    std::vector<uint8_t> nextTypes;
    std::vector<int> nextMaskIndices;
    std::vector<uint8_t> nextChildMasks;
    nextMortons.reserve(n);
    nextTypes.reserve(n);
    nextMaskIndices.reserve(n);
    nextChildMasks.reserve(n);

    size_t i = 0;
    while (i < n) {
      const uint32_t parentMorton = (*curMortons)[i] / 8;
      uint8_t childMask = 0;
      bool allSolid = true;
      int childCount = 0;

      // Scan all consecutive entries that share this parent
      while (i < n && (*curMortons)[i] / 8 == parentMorton) {
        const uint32_t octant = (*curMortons)[i] % 8;
        childMask |= static_cast<uint8_t>(1u << octant);
        if ((*curTypes)[i] != static_cast<uint8_t>(BlockType::Solid)) {
          allSolid = false;
        }
        childCount++;
        i++;
      }

      if (allSolid && childCount == 8) {
        nextMortons.push_back(parentMorton);
        nextTypes.push_back(static_cast<uint8_t>(BlockType::Solid));
        nextMaskIndices.push_back(-1);
        nextChildMasks.push_back(0);
      } else {
        nextMortons.push_back(parentMorton);
        nextTypes.push_back(static_cast<uint8_t>(BlockType::Mixed));
        nextMaskIndices.push_back(-1);
        nextChildMasks.push_back(childMask);
      }
    }

    // Update current level references
    sortedMortons = std::move(nextMortons);
    sortedTypes = std::move(nextTypes);
    sortedMaskIndices = std::move(nextMaskIndices);
    curChildMasks = std::move(nextChildMasks);

    curMortons = &sortedMortons;
    curTypes = &sortedTypes;
    curMaskIndices = &sortedMaskIndices;

    // Break when the tree is empty or has converged to a single root at Morton 0
    if (curMortons->empty() || (curMortons->size() == 1 && (*curMortons)[0] == 0)) {
      actualDepth = level + 1;
      break;
    }
  }

  // Save the root level
  levels.push_back({*curMortons, *curTypes, *curMaskIndices, curChildMasks});

  // Flatten tree to Laine-Karras format using wave-based BFS
  const auto& rootLevel = levels.back();

  if (rootLevel.mortons.empty()) {
    return {gridBounds, sceneBounds, voxelResolution, 4, actualDepth, 0, 0, {}, {}};
  }

  // Upper bound on total nodes
  size_t maxNodes = 0;
  for (const auto& level : levels) {
    maxNodes += level.mortons.size();
  }

  std::vector<uint32_t> nodes(maxNodes);
  std::vector<uint32_t> leafDataList;
  leafDataList.reserve(maxNodes * 2);
  int numInteriorNodes = 0;
  int numMixedLeaves = 0;
  size_t emitPos = 0;

  // BFS wave as parallel arrays
  std::vector<int> waveLi;     // Level index
  std::vector<size_t> waveIi;  // Index within level

  // Initialize wave with root level entries
  const int rootLi = static_cast<int>(levels.size()) - 1;
  for (size_t i = 0; i < rootLevel.mortons.size(); i++) {
    waveLi.push_back(rootLi);
    waveIi.push_back(i);
  }

  // Reusable arrays for tracking interior nodes within each wave
  std::vector<size_t> intPos;
  std::vector<int> intLi;
  std::vector<size_t> intIi;
  std::vector<uint8_t> intMask;

  while (!waveLi.empty()) {
    intPos.clear();
    intLi.clear();
    intIi.clear();
    intMask.clear();

    // Emit all nodes in this wave
    for (size_t w = 0; w < waveLi.size(); w++) {
      const int li = waveLi[w];
      const size_t ii = waveIi[w];
      const auto& level = levels[static_cast<size_t>(li)];
      const uint8_t type = level.types[ii];

      // A node is a leaf if it's Solid (at any level) or if it's at level 0
      const bool isLeaf = (type == static_cast<uint8_t>(BlockType::Solid)) || (li == 0);

      if (isLeaf) {
        if (type == static_cast<uint8_t>(BlockType::Solid)) {
          nodes[emitPos] = SOLID_LEAF_MARKER;
        } else {
          // Mixed leaf — store index into leafData
          const int maskIdx = level.maskIndices[ii];
          const uint32_t leafDataIndex = static_cast<uint32_t>(leafDataList.size() / 2);
          leafDataList.push_back(mixed.masks[static_cast<size_t>(maskIdx) * 2]);
          leafDataList.push_back(mixed.masks[static_cast<size_t>(maskIdx) * 2 + 1]);
          nodes[emitPos] = leafDataIndex & 0x00FFFFFFu;
          numMixedLeaves++;
        }
      } else {
        // Interior node — record position for backfill after wave
        intPos.push_back(emitPos);
        intLi.push_back(li);
        intIi.push_back(ii);
        intMask.push_back(level.childMasks[ii]);
        numInteriorNodes++;
        nodes[emitPos] = 0;  // Placeholder
      }
      emitPos++;
    }

    // Build next wave from children of interior nodes
    std::vector<int> nextWaveLi;
    std::vector<size_t> nextWaveIi;
    size_t nextChildStart = emitPos;

    for (size_t j = 0; j < intPos.size(); j++) {
      const uint8_t childMask = intMask[j];
      const int childCount = absl::popcount(childMask);

      // Encode interior node: mask in high byte, baseOffset in low 24 bits
      nodes[intPos[j]] =
          (static_cast<uint32_t>(childMask) << 24) | (static_cast<uint32_t>(nextChildStart) & 0x00FFFFFFu);

      // Find children in the level below using binary search
      const int childLi = intLi[j] - 1;
      const auto& childLevel = levels[static_cast<size_t>(childLi)];
      const uint32_t myMorton = levels[static_cast<size_t>(intLi[j])].mortons[intIi[j]];
      const uint32_t childMortonBase = myMorton * 8;
      const uint32_t childMortonEnd = childMortonBase + 8;
      const auto& childMortons = childLevel.mortons;

      // Binary search for first child with morton >= childMortonBase
      size_t lo = 0;
      size_t hi = childMortons.size();
      while (lo < hi) {
        const size_t mid = (lo + hi) >> 1;
        if (childMortons[mid] < childMortonBase)
          lo = mid + 1;
        else
          hi = mid;
      }

      // Collect all children in morton order
      while (lo < childMortons.size() && childMortons[lo] < childMortonEnd) {
        nextWaveLi.push_back(childLi);
        nextWaveIi.push_back(lo);
        lo++;
      }

      nextChildStart += static_cast<size_t>(childCount);
    }

    waveLi = std::move(nextWaveLi);
    waveIi = std::move(nextWaveIi);
  }

  return {gridBounds,
          sceneBounds,
          voxelResolution,
          4,
          actualDepth,
          numInteriorNodes,
          numMixedLeaves,
          std::vector<uint32_t>(nodes.begin(), nodes.begin() + static_cast<ptrdiff_t>(emitPos)),
          std::move(leafDataList)};
}

}  // namespace splat
