/**
 * @file splat/models/sog.h
 * @brief SOG model types.
 *
 */
 
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace splat {

/**
 * @struct Meta
 * @brief Metadata structure for Gaussian Splatting data representation
 *
 * The Meta struct contains comprehensive metadata for Gaussian Splatting data,
 * including version information, asset generation details, and file organization
 * for various data components (means, scales, rotations, spherical harmonics).
 * It supports parsing from and encoding to JSON format.
 */
struct Meta {
  int version;  ///< Version number of the metadata format
  int count;    ///< Total number of Gaussian splats

  /**
   * @struct AssetInfo
   * @brief Information about the generating asset/application
   */
  struct {
    std::string generator;  ///< Name/identifier of the generating software/tool
  } asset;

  /**
   * @struct MeansInfo
   * @brief Information about Gaussian mean positions
   *
   * Stores bounding information and file references for Gaussian means (positions).
   */
  struct {
    std::vector<float> mins;         ///< Minimum bounds for mean positions (x, y, z)
    std::vector<float> maxs;         ///< Maximum bounds for mean positions (x, y, z)
    std::vector<std::filesystem::path> files;  ///< File paths containing mean position data
  } means;

  /**
   * @struct ScalesInfo
   * @brief Information about Gaussian scale factors
   *
   * Stores quantization codebook and file references for Gaussian scales.
   */
  struct {
    std::vector<float> codebook;     ///< Quantization codebook for scale values
    std::vector<std::filesystem::path> files;  ///< File paths containing scale data
  } scales;

  /**
   * @struct QuatsInfo
   * @brief Information about Gaussian rotations (quaternions)
   */
  struct {
    std::vector<std::filesystem::path> files;  ///< File paths containing quaternion rotation data
  } quats;

  /**
   * @struct SH0Info
   * @brief Information about 0th order spherical harmonics (SH0)
   *
   * Represents the base color/opacity components (typically RGBA).
   */
  struct {
    std::vector<float> codebook;     ///< Quantization codebook for SH0 values
    std::vector<std::filesystem::path> files;  ///< File paths containing SH0 data
  } sh0;

  /**
   * @struct SHN
   * @brief Information about higher order spherical harmonics (SH > 0)
   *
   * Represents higher frequency color/lighting components.
   */
  struct SHN {
    int count;                       ///< Number of SH bands (degrees) stored
    int bands;                       ///< Number of spherical harmonic bands represented
    std::vector<float> codebook;     ///< Quantization codebook for SHN values
    std::vector<std::filesystem::path> files;  ///< File paths containing SHN data
  };

  /**
   * @brief Optional higher order spherical harmonics information
   *
   * If present, indicates the data contains spherical harmonics beyond SH0.
   * Absent if only SH0 (base color) is used.
   */
  std::optional<SHN> shN;

  /**
   * @brief Parse metadata from JSON byte array
   * @param json JSON data as vector of bytes (UTF-8 encoded)
   * @return Meta object populated from JSON data
   * @throws std::runtime_error if JSON parsing fails or required fields are missing
   *
   * Expected JSON structure:
   * {
   *   "version": int,
   *   "count": int,
   *   "asset": {"generator": "string"},
   *   "means": {"mins": [float, float, float], "maxs": [float, float, float], "files": ["string"]},
   *   "scales": {"codebook": [float...], "files": ["string"]},
   *   "quats": {"files": ["string"]},
   *   "sh0": {"codebook": [float...], "files": ["string"]},
   *   "shN": {"count": int, "bands": int, "codebook": [float...], "files": ["string"]} // optional
   * }
   */
  static Meta parseFromJson(const std::vector<uint8_t>& json);

  /**
   * @brief Encode metadata to JSON string
   * @return JSON string representation of the metadata
   *
   * Converts all metadata fields to a JSON object following the same structure
   * expected by parseFromJson. The output is compact (no whitespace) for efficiency.
   */
  std::string encodeToJson() const;
};

}  // namespace splat
