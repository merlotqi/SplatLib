/**
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
 */

#include <splat/readers/ksplat_reader.h>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace splat {

struct CompressionConfig {
  size_t centerBytes;
  size_t scaleBytes;
  size_t rotationBytes;
  size_t colorBytes;
  size_t harmonicsBytes;
  size_t scaleStartByte;
  size_t rotationStartByte;
  size_t colorStartByte;
  size_t harmonicsStartByte;
  uint32_t scaleQuantRange;
};

static const CompressionConfig COMPRESSION_MODES[] = {
    // Mode 0: Full precision
    {12, 12, 16, 4, 4, 12, 24, 40, 44, 1},
    // Mode 1: Half precision / Quantized position
    {
        6, 6, 8, 4, 2, 6, 12, 20, 24,
        32767  // 2^15 - 1
    },
    // Mode 2: Byte harmonics / Half scale/rotation / Quantized position
    {6, 6, 8, 4, 1, 6, 12, 20, 24, 32767}};

static const size_t HARMONICS_COMPONENT_COUNT[] = {0, 9, 24, 45};

static float decodeFloat16(uint16_t encoded) {
  const uint32_t signBit = (encoded >> 15) & 1;
  const uint32_t exponent = (encoded >> 10) & 0x1f;
  const uint32_t mantissa = encoded & 0x3ff;

  if (exponent == 0) {
    if (mantissa == 0) return signBit ? -0.0f : 0.0f;

    // Denormalized number
    uint32_t m = mantissa;
    int exp = -14;
    while (!(m & 0x400)) {
      m <<= 1;
      exp--;
    }
    m &= 0x3ff;
    const uint32_t finalExp = exp + 127;
    const uint32_t finalMantissa = m << 13;
    uint32_t bits = (signBit << 31) | (finalExp << 23) | finalMantissa;
    return *reinterpret_cast<float*>(&bits);
  }

  if (exponent == 0x1f) {
    return mantissa == 0 ? (signBit ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity())
                         : std::numeric_limits<float>::quiet_NaN();
  }

  const uint32_t finalExp = exponent - 15 + 127;
  const uint32_t finalMantissa = mantissa << 13;
  uint32_t bits = (signBit << 31) | (finalExp << 23) | finalMantissa;
  return *reinterpret_cast<float*>(&bits);
}

std::unique_ptr<DataTable> readKsplat(const std::string& filename) {
  const size_t totalSize = fs::file_size(filename);

  // Load complete file
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  static constexpr auto MAIN_HEADER_SIZE = 4096ULL;
  static constexpr auto SECTION_HEADER_SIZE = 1024ULL;

  if (totalSize < MAIN_HEADER_SIZE) {
    throw std::runtime_error("File too small to be valid .ksplat format.");
  }

  auto getUint32 = [](const uint8_t* data, size_t offset) {
    uint32_t v;
    std::memcpy(&v, data + offset, sizeof(v));
    return v;
  };

  auto getUint16 = [](const uint8_t* data, size_t offset) {
    uint16_t v;
    std::memcpy(&v, data + offset, sizeof(v));
    return v;
  };

  auto getFloat32 = [](const uint8_t* data, size_t offset) {
    float v;
    std::memcpy(&v, data + offset, sizeof(v));
    return v;
  };

  // parse main header
  std::vector<uint8_t> mainHeaderData(MAIN_HEADER_SIZE);
  if (!file.read(reinterpret_cast<char*>(mainHeaderData.data()), MAIN_HEADER_SIZE)) {
    throw std::runtime_error("File too small or failed to read main header.");
  }
  const uint8_t majorVersion = mainHeaderData[0];
  const uint8_t minorVersion = mainHeaderData[1];
  if (majorVersion != 0 || minorVersion < 1) {
    throw std::runtime_error("Unsupported version " + std::to_string(majorVersion) + "." +
                             std::to_string(minorVersion));
  }

  const uint32_t maxSections = getUint32(mainHeaderData.data(), 4);
  const uint32_t numSplats = getUint32(mainHeaderData.data(), 16);
  const uint16_t compressionMode = getUint16(mainHeaderData.data(), 20);
  if (compressionMode > 2) {
    throw std::runtime_error("Invalid compression mode: " + std::to_string(compressionMode));
  }

  const float minHarmonicsValue = getFloat32(mainHeaderData.data(), 36);
  const float maxHarmonicsValue = getFloat32(mainHeaderData.data(), 40);
  if (numSplats == 0) {
    throw std::runtime_error("Invalid .ksplat file: file is empty");
  }

  // First pass: scan all sections to find maximum harmonics degree
  uint16_t maxHarmonicsDegree = 0;
  for (uint32_t sectionIdx = 0; sectionIdx < maxSections; sectionIdx++) {
    const size_t sectionHeaderOffset = MAIN_HEADER_SIZE + sectionIdx * SECTION_HEADER_SIZE;
    (void)sectionHeaderOffset;
    std::vector<uint8_t> sectionHeaderData(SECTION_HEADER_SIZE);
    if (!file.read(reinterpret_cast<char*>(sectionHeaderData.data()), SECTION_HEADER_SIZE)) {
      throw std::runtime_error("File too small or failed to read section header.");
    }

    const uint32_t sectionSplatCount = getUint32(sectionHeaderData.data(), 0);
    if (sectionSplatCount == 0) continue;

    const uint16_t harmonicsDegree = getUint16(sectionHeaderData.data(), 40);
    maxHarmonicsDegree = std::max(maxHarmonicsDegree, harmonicsDegree);
  }

  // Initialize data storage with base columns
  std::vector<Column> columns = {// Position
                                 {"x", std::vector<float>(numSplats, 0.0f)},
                                 {"y", std::vector<float>(numSplats, 0.0f)},
                                 {"z", std::vector<float>(numSplats, 0.0f)},

                                 // Scale (stored as linear in .splat, convert to log for internal use)
                                 {"scale0", std::vector<float>(numSplats, 0.0f)},
                                 {"scale1", std::vector<float>(numSplats, 0.0f)},
                                 {"scale2", std::vector<float>(numSplats, 0.0f)},

                                 // Color/opacity
                                 {"f_dc_0", std::vector<float>(numSplats, 0.0f)},  // Red
                                 {"f_dc_1", std::vector<float>(numSplats, 0.0f)},  // Green
                                 {"f_dc_2", std::vector<float>(numSplats, 0.0f)},  // Blue
                                 {"opacity", std::vector<float>(numSplats, 0.0f)},

                                 // Rotation quaternion
                                 {"rot_0", std::vector<float>(numSplats, 0.0f)},
                                 {"rot_1", std::vector<float>(numSplats, 0.0f)},
                                 {"rot_2", std::vector<float>(numSplats, 0.0f)},
                                 {"rot_3", std::vector<float>(numSplats, 0.0f)}};

  // Add spherical harmonics columns based on maximum degree found
  const size_t maxHarmonicsComponentCount = HARMONICS_COMPONENT_COUNT[maxHarmonicsDegree];
  for (size_t i = 0; i < maxHarmonicsComponentCount; i++) {
    columns.push_back({"f_rest_" + std::to_string(i), std::vector<float>(numSplats)});
  }

  // clang-format off
  const auto&[        
        centerBytes,
        scaleBytes,
        rotationBytes,
        colorBytes,
        harmonicsBytes,
        scaleStartByte,
        rotationStartByte,
        colorStartByte,
        harmonicsStartByte,
        scaleQuantRange] = COMPRESSION_MODES[compressionMode];
  // clang-format on

  const auto currentSectionDataOffset = MAIN_HEADER_SIZE + maxSections * SECTION_HEADER_SIZE;
  size_t splatIndex = 0;

  // Process each section
  for (size_t sectionIdx = 0; sectionIdx < maxSections; ++sectionIdx) {
    const auto sectionHeaderOffset = MAIN_HEADER_SIZE + sectionIdx * SECTION_HEADER_SIZE;
    file.seekg(sectionHeaderOffset, std::ios::beg);

    std::vector<uint8_t> sectionHeader;
    if (!file.read(reinterpret_cast<char*>(sectionHeader.data()), SECTION_HEADER_SIZE)) {
      continue;  // Section header is invalid/missing, stop processing sections
    }

    const auto sectionSplatCount = getUint32(sectionHeader.data(), 0);
    const auto maxSectionSplats = getUint32(sectionHeader.data(), 4);
    const auto bucketCapacity = getUint32(sectionHeader.data(), 8);
    const auto bucketCount = getUint32(sectionHeader.data(), 12);
    const auto spatialBlockSize = getFloat32(sectionHeader.data(), 16);
    const auto bucketStorageSize = getUint16(sectionHeader.data(), 20);
    const auto quantizationRange = getUint32(sectionHeader.data(), 24) || scaleQuantRange;
    const auto fullBuckets = getUint32(sectionHeader.data(), 32);
    const auto partialBuckets = getUint32(sectionHeader.data(), 36);
    const auto harmonicsDegree = getUint16(sectionHeader.data(), 40);

    // Calculate layout
    const uint32_t fullBucketSplats = fullBuckets * bucketCapacity;
    const size_t partialBucketMetaSize = partialBuckets * 4;
    const size_t totalBucketStorageSize = bucketStorageSize * bucketCount + partialBucketMetaSize;
    const size_t harmonicsComponentCount = HARMONICS_COMPONENT_COUNT[harmonicsDegree];
    const size_t bytesPerSplat =
        centerBytes + scaleBytes + rotationBytes + colorBytes + harmonicsComponentCount * harmonicsBytes;
    const size_t sectionDataSize = bytesPerSplat * maxSectionSplats;

    // Calculate decompression parameters
    const float positionScale = spatialBlockSize / 2.0f / quantizationRange;

    // Get bucket centers
    const auto bucketCentersOffset = currentSectionDataOffset + partialBucketMetaSize;
    (void)bucketCentersOffset;
    std::vector<float> bucketCenters(bucketCount * 3);
    if (!file.read(reinterpret_cast<char*>(bucketCenters.data()), bucketCount * 3 * sizeof(float))) {
      throw std::runtime_error("Failed to read bucket centers");
    }
    // Get partial bucket sizes
    std::vector<uint32_t> partialBucketSizes(partialBuckets);
    if (!file.read(reinterpret_cast<char*>(partialBucketSizes.data()), partialBucketSizes.size() * sizeof(uint32_t))) {
      throw std::runtime_error("Failed to read partial bucket sizes");
    }
    // Get splat data
    std::vector<uint8_t> splatData(sectionDataSize);
    if (!file.read(reinterpret_cast<char*>(splatData.data()), splatData.size())) {
      throw std::runtime_error("Failed to read splat data");
    }

    auto decodeHarmonics = [&](size_t offset, size_t component) {
      switch (compressionMode) {
        case 0:
          return getFloat32(splatData.data(), offset + harmonicsStartByte + component * 4);
        case 1:
          return decodeFloat16(getUint16(splatData.data(), offset + harmonicsStartByte + component * 2));
        case 2: {
          uint8_t normalized = splatData[offset + harmonicsStartByte + component];
          return minHarmonicsValue + (float)normalized / 255.0f * (maxHarmonicsValue - minHarmonicsValue);
        }
        default:
          return 0.f;
      }
    };

    // Track partial bucket processing
    uint32_t currentPartialBucket = fullBuckets;
    uint32_t currentPartialBase = fullBucketSplats;

    // Process splats in this section
    for (size_t splatIdx = 0; splatIdx < sectionSplatCount; ++splatIdx) {
      const size_t splatByteOffset = splatIdx * bytesPerSplat;

      // Determine which bucket this splat belongs to
      uint32_t bucketIdx;
      if (splatIdx < fullBucketSplats) {
        bucketIdx = std::floor(splatIdx / bucketCapacity);
      } else {
        const auto currentBucketSize = partialBucketSizes[currentPartialBucket - fullBuckets];
        if (splatIdx >= currentPartialBase + currentBucketSize) {
          currentPartialBucket++;
          currentPartialBase += currentBucketSize;
        }
        bucketIdx = currentPartialBucket;
      }

      // Decode position
      float x, y, z;
      if (compressionMode == 0) {
        x = getFloat32(splatData.data(), splatByteOffset + 0);
        y = getFloat32(splatData.data(), splatByteOffset + 4);
        z = getFloat32(splatData.data(), splatByteOffset + 8);
      } else {
        x = (getUint16(splatData.data(), splatByteOffset + 0) - quantizationRange) * positionScale +
            bucketCenters[bucketIdx * 3];
        y = (getUint16(splatData.data(), splatByteOffset + 2) - quantizationRange) * positionScale +
            bucketCenters[bucketIdx * 3 + 1];
        z = (getUint16(splatData.data(), splatByteOffset + 4) - quantizationRange) * positionScale +
            bucketCenters[bucketIdx * 3 + 2];
      }

      // Decode scales
      float scaleX, scaleY, scaleZ;
      if (compressionMode == 0) {
        scaleX = getFloat32(splatData.data(), splatByteOffset + scaleStartByte + 0);
        scaleY = getFloat32(splatData.data(), splatByteOffset + scaleStartByte + 4);
        scaleZ = getFloat32(splatData.data(), splatByteOffset + scaleStartByte + 8);
      } else {
        scaleX = decodeFloat16(getUint16(splatData.data(), splatByteOffset + scaleStartByte + 0));
        scaleY = decodeFloat16(getUint16(splatData.data(), splatByteOffset + scaleStartByte + 2));
        scaleZ = decodeFloat16(getUint16(splatData.data(), splatByteOffset + scaleStartByte + 4));
      }

      // Decode rotation quaternion
      float rot0, rot1, rot2, rot3;
      if (compressionMode == 0) {
        rot0 = getFloat32(splatData.data(), splatByteOffset + rotationStartByte + 0);
        rot1 = getFloat32(splatData.data(), splatByteOffset + rotationStartByte + 4);
        rot2 = getFloat32(splatData.data(), splatByteOffset + rotationStartByte + 8);
        rot3 = getFloat32(splatData.data(), splatByteOffset + rotationStartByte + 12);
      } else {
        rot0 = decodeFloat16(getUint16(splatData.data(), splatByteOffset + rotationStartByte + 0));
        rot1 = decodeFloat16(getUint16(splatData.data(), splatByteOffset + rotationStartByte + 2));
        rot2 = decodeFloat16(getUint16(splatData.data(), splatByteOffset + rotationStartByte + 4));
        rot3 = decodeFloat16(getUint16(splatData.data(), splatByteOffset + rotationStartByte + 6));
      }

      // Decode color and opacity
      uint8_t red = splatData[splatByteOffset + colorStartByte + 0];
      uint8_t green = splatData[splatByteOffset + colorStartByte + 1];
      uint8_t blue = splatData[splatByteOffset + colorStartByte + 2];
      uint8_t opacity = splatData[splatByteOffset + colorStartByte + 3];

      // store position
      columns[0].setValue<float>(splatIdx, x);
      columns[1].setValue<float>(splatIdx, y);
      columns[2].setValue<float>(splatIdx, z);

      // Store scale (convert from linear in .ksplat to log scale for internal use)
      columns[3].setValue<float>(splatIdx, scaleX > 0 ? logf(scaleX) : -10.0f);
      columns[4].setValue<float>(splatIdx, scaleY > 0 ? logf(scaleY) : -10.0f);
      columns[5].setValue<float>(splatIdx, scaleZ > 0 ? logf(scaleZ) : -10.0f);

      // Store color (convert from uint8 back to spherical harmonics)
      static constexpr auto SH_C0 = 0.28209479177387814;
      columns[6].setValue<float>(splatIdx, (red / 255.0f - 0.5f) / SH_C0);
      columns[7].setValue<float>(splatIdx, (green / 255.0f - 0.5f) / SH_C0);
      columns[8].setValue<float>(splatIdx, (blue / 255.0f - 0.5f) / SH_C0);

      // Store opacity (convert from uint8 to float and apply inverse sigmoid)
      static constexpr auto epsilon = 1e-6f;
      const auto normalizedOpacity = std::max(epsilon, std::min(1.0f - epsilon, opacity / 255.0f));
      columns[9].setValue<float>(splatIdx, logf(normalizedOpacity / (1.0 - normalizedOpacity)));

      // Store quaternion
      columns[10].setValue<float>(splatIdx, rot0);
      columns[11].setValue<float>(splatIdx, rot1);
      columns[12].setValue<float>(splatIdx, rot2);
      columns[13].setValue<float>(splatIdx, rot3);

      // Store spherical harmonics
      const size_t baseColumnIndex = 14;
      for (size_t i = 0; i < harmonicsComponentCount; ++i) {
        size_t channel;
        size_t coeff;
        // Matching the TypeScript SH packing logic
        if (i < 9) {
          channel = i / 3;
          coeff = i % 3;
        } else if (i < 24) {
          channel = (i - 9) / 5;
          coeff = (i - 9) % 5 + 3;
        } else {
          channel = (i - 24) / 7;
          coeff = (i - 24) % 7 + 8;
        }

        const size_t col = channel * (maxHarmonicsComponentCount / 3) + coeff;
        columns[baseColumnIndex + col].setValue<float>(splatIdx, decodeHarmonics(splatByteOffset, i));
      }

      splatIdx++;
    }
    file.seekg(static_cast<std::streamsize>(file.tellg()) + sectionDataSize + totalBucketStorageSize);
  }

  if (splatIndex != numSplats) {
    throw std::runtime_error("Splat count mismatch: expected " + std::to_string(numSplats) + ", processed " +
                             std::to_string(splatIndex));
  }
  return std::make_unique<DataTable>(columns);
}

}  // namespace splat
