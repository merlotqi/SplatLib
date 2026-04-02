/**
 * @file splat/models/lcc.h
 * @brief LoD / LCC structures for Level-of-Detail Gaussian splatting
 *
 * Defines data structures for LCC (Layered Compressed Cloud) format,
 * which supports multi-level-of-detail streaming of Gaussian splat data
 * with quadtree spatial partitioning.
 */
 
#pragma once

#include <Eigen/Dense>
#include <map>

namespace splat {

/**
 * @brief Level-of-detail data block in data.bin
 */
struct LccLod {
  int32_t points;  ///< Number of splats in this LOD level
  int64_t offset;  ///< Byte offset to splat data
  int32_t size;    ///< Data block size in bytes
};

/**
 * @brief Quadtree unit info with spatial index and LOD levels
 *
 * The scene uses a quadtree for spatial partitioning, with each unit
 * having its own xy index (starting from 0) and multiple layers of LOD data.
 */
struct LccUnitInfo {
  int16_t x;                  ///< X index in quadtree grid
  int16_t y;                  ///< Y index in quadtree grid
  std::vector<LccLod> lods;   ///< LOD data blocks for this unit
};

/**
 * @brief Compression parameters for splat and environment attributes
 *
 * Stores min/max bounds used for dequantizing compressed scale and
 * spherical harmonic coefficient data.
 */
struct CompressInfo {
  Eigen::Vector3f scaleMin;     ///< Minimum scale values (splat)
  Eigen::Vector3f scaleMax;     ///< Maximum scale values (splat)
  Eigen::Vector3f shMin;        ///< Minimum SH values (splat)
  Eigen::Vector3f shMax;        ///< Maximum SH values (splat)
  Eigen::Vector3f envScaleMin;  ///< Minimum scale values (environment)
  Eigen::Vector3f envScaleMax;  ///< Maximum scale values (environment)
  Eigen::Vector3f envShMin;     ///< Minimum SH values (environment)
  Eigen::Vector3f envShMax;     ///< Maximum SH values (environment)
};

/**
 * @brief Parameters for converting LCC data to GSplatData
 */
struct LccParam {
  int totalSplats;                ///< Total number of splats across all units
  int targetLod;                  ///< Target LOD level for loading
  CompressInfo compressInfo;      ///< Compression parameters
  std::vector<LccUnitInfo> unitInfos;  ///< Quadtree unit information
  std::string dataFile;          ///< Path to splat data file
  std::string shFile;            ///< Path to SH data file
};

/**
 * @brief Processing context for a single LCC unit
 *
 * Holds intermediate state when processing a quadtree unit's data,
 * including properties and compression parameters.
 */
struct ProcessUnitContext {
  LccUnitInfo info;                                        ///< Unit info
  int targetLod;                                           ///< Target LOD level
  std::string dataFile;                                    ///< Path to data file
  std::string shFile;                                      ///< Path to SH file
  CompressInfo compressInfo;                               ///< Compression parameters
  float propertyOffset;                                    ///< Property value offset
  std::map<std::string, std::vector<float>> properties;   ///< Named float properties
  std::vector<float> properties_f_rest;                   ///< Rest frame properties
};

}  // namespace splat
