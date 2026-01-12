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

#include <splat/io/splat_reader.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace splat {

// each splat is 32 bytes
static constexpr auto BYTES_PER_SPLAT = 32;

static float readFloatLE(const std::vector<uint8_t>& data, size_t offset) {
  if (offset + sizeof(float) > data.size()) {
    throw std::out_of_range("Offset is out of bounds for reading a float.");
  }

  float value;
  const uint8_t* src = data.data() + offset;

  std::memcpy(&value, src, sizeof(float));

  return value;
}

static uint8_t readUInt8(const std::vector<uint8_t>& data, size_t offset) { return data[offset]; }

std::unique_ptr<DataTable> readSplat(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::in);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  const size_t fileSize = fs::file_size(filename);
  if (fileSize % BYTES_PER_SPLAT != 0) {
    throw std::runtime_error("Invalid .splat file: file size is not a multiple of 32 bytes");
  }

  const size_t numSplats = fileSize / BYTES_PER_SPLAT;
  if (numSplats == 0) {
    throw std::runtime_error("Invalid .splat file: file is empty");
  }

  // Create columns for the standard Gaussian splat data
  std::vector<Column> columns = {
      // Position
      {"x", std::vector<float>(numSplats, 0.0f)},
      {"y", std::vector<float>(numSplats, 0.0f)},
      {"z", std::vector<float>(numSplats, 0.0f)},

      // Scale (stored as linear in .splat, convert to log for internal use)
      {"scale_0", std::vector<float>(numSplats, 0.0f)},
      {"scale_1", std::vector<float>(numSplats, 0.0f)},
      {"scale_2", std::vector<float>(numSplats, 0.0f)},

      // Color/opacity
      {"f_dc_0", std::vector<float>(numSplats, 0.0f)},  // red
      {"f_dc_1", std::vector<float>(numSplats, 0.0f)},  // green
      {"f_dc_2", std::vector<float>(numSplats, 0.0f)},  // blue
      {"opacity", std::vector<float>(numSplats, 0.0f)},

      // Rotation quaternion
      {"rot_0", std::vector<float>(numSplats, 0.0f)},
      {"rot_1", std::vector<float>(numSplats, 0.0f)},
      {"rot_2", std::vector<float>(numSplats, 0.0f)},
      {"rot_3", std::vector<float>(numSplats, 0.0f)},
  };

  // Read data in chunks
  const size_t chunkSize = 1024;
  const size_t numChunks = ceil(static_cast<double>(numSplats) / chunkSize);
  std::vector<uint8_t> chunkData(chunkSize * BYTES_PER_SPLAT);

  for (size_t c = 0; c < numChunks; ++c) {
    const auto numRows = std::min(chunkSize, numSplats - c * chunkSize);
    const size_t bytesToRead = numRows * BYTES_PER_SPLAT;

    file.read(reinterpret_cast<char*>(chunkData.data()), bytesToRead);
    const size_t bytesRead = file.gcount();
    if (bytesRead != bytesToRead) {
      throw std::runtime_error("Failed to read expected amount of data from .splat file");
    }

    // Parse each splat in the chunk
    for (size_t r = 0; r < numRows; ++r) {
      const auto splatIndex = c * chunkSize + r;
      const auto offset = r * BYTES_PER_SPLAT;

      // Read position (3 × float32)
      const float x = readFloatLE(chunkData, offset + 0);
      const float y = readFloatLE(chunkData, offset + 4);
      const float z = readFloatLE(chunkData, offset + 8);

      // Read scale (3 × float32)
      const float scaleX = readFloatLE(chunkData, offset + 12);
      const float scaleY = readFloatLE(chunkData, offset + 16);
      const float scaleZ = readFloatLE(chunkData, offset + 20);

      // Read color and opacity (4 × uint8)
      const uint8_t red = readUInt8(chunkData, offset + 24);
      const uint8_t green = readUInt8(chunkData, offset + 25);
      const uint8_t blue = readUInt8(chunkData, offset + 26);
      const uint8_t opacity = readUInt8(chunkData, offset + 27);

      // Read rotation quaternion (4 × uint8)
      const uint8_t rot0 = readUInt8(chunkData, offset + 28);
      const uint8_t rot1 = readUInt8(chunkData, offset + 29);
      const uint8_t rot2 = readUInt8(chunkData, offset + 30);
      const uint8_t rot3 = readUInt8(chunkData, offset + 31);

      // store position
      columns[0].setValue<float>(splatIndex, x);
      columns[1].setValue<float>(splatIndex, y);
      columns[2].setValue<float>(splatIndex, z);

      // store scale (convert to log scale)
      columns[3].setValue<float>(splatIndex, logf(scaleX));
      columns[4].setValue<float>(splatIndex, logf(scaleY));
      columns[5].setValue<float>(splatIndex, logf(scaleZ));

      // Store color (convert from uint8 back to spherical harmonics)
      static constexpr auto SH_C0 = 0.28209479177387814;
      columns[6].setValue<float>(splatIndex, ((red / 255.0 - 0.5) / SH_C0));
      columns[7].setValue<float>(splatIndex, ((green / 255.0 - 0.5) / SH_C0));
      columns[8].setValue<float>(splatIndex, ((blue / 255.0 - 0.5) / SH_C0));

      // Store opacity (convert from uint8 to float and apply inverse sigmoid)
      static constexpr auto epsilon = 1.0e-6;
      const auto normalizedOpacity = std::max(epsilon, std::min(1.0 - epsilon, opacity / 255.0));
      columns[9].setValue<float>(splatIndex, logf(normalizedOpacity / (1.0 - normalizedOpacity)));

      // Store rotation quaternion (convert from uint8 [0,255] to float [-1,1] and normalize)
      const auto rot0Norm = (rot0 / 255.0) * 2.0 - 1.0;
      const auto rot1Norm = (rot1 / 255.0) * 2.0 - 1.0;
      const auto rot2Norm = (rot2 / 255.0) * 2.0 - 1.0;
      const auto rot3Norm = (rot3 / 255.0) * 2.0 - 1.0;

      // Normalize quaternion
      const auto length =
          std::sqrt(rot0Norm * rot0Norm + rot1Norm * rot1Norm + rot2Norm * rot2Norm + rot3Norm * rot3Norm);
      if (length > 0.0) {
        columns[10].setValue<float>(splatIndex, rot0Norm / length);
        columns[11].setValue<float>(splatIndex, rot1Norm / length);
        columns[12].setValue<float>(splatIndex, rot2Norm / length);
        columns[13].setValue<float>(splatIndex, rot3Norm / length);
      } else {
        // Default to identity quaternion if invalid
        columns[10].setValue<float>(splatIndex, 0.0f);
        columns[11].setValue<float>(splatIndex, 0.0f);
        columns[12].setValue<float>(splatIndex, 0.0f);
        columns[13].setValue<float>(splatIndex, 1.0f);
      }
    }
  }
  return std::make_unique<DataTable>(columns);
}

}  // namespace splat
