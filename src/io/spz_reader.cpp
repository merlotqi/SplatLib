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

#include <splat/io/spz_reader.h>
#include <zlib.h>

#include <fstream>

namespace splat {

static const size_t SPZ_HEADER_SIZE = 16;
const float SH_C0_2 = 0.15f;
const size_t HARMONICS_COMPONENT_COUNT[] = {0, 9, 24, 45};

static float inverseConvertColorFromSPZ(float y) { return (y / 255.0f - 0.5f) / SH_C0_2; }

static std::vector<uint8_t> decompressGZIP(const std::vector<uint8_t>& compressedData) {
  if (compressedData.size() < 18) throw std::runtime_error("Buffer too small to be GZip");

  z_stream zs;
  std::memset(&zs, 0, sizeof(z_stream));

  if (inflateInit2(&zs, 16 + MAX_WBITS) != Z_OK) {
    throw std::runtime_error("inflateInit2 failed");
  }

  zs.next_in = const_cast<Bytef*>(compressedData.data());
  zs.avail_in = static_cast<uInt>(compressedData.size());

  int ret;
  std::vector<uint8_t> outBuffer;
  uint8_t temp_buf[10240] = {0};
  do {
    zs.next_out = temp_buf;
    zs.avail_out = sizeof(temp_buf);

    ret = inflate(&zs, Z_NO_FLUSH);

    if (outBuffer.size() < zs.total_out) {
      size_t decompressedChunkSize = sizeof(temp_buf) - zs.avail_out;
      outBuffer.insert(outBuffer.end(), temp_buf, temp_buf + decompressedChunkSize);
    }

  } while (ret == Z_OK);

  inflateEnd(&zs);

  if (ret != Z_STREAM_END) {
    throw std::runtime_error("GZip decompression failed: " + std::to_string(ret));
  }

  return outBuffer;
}

static int32_t getFixed24(const std::vector<uint8_t>& buffer, size_t elementIndex, size_t memberIndex) {
  const size_t stride = 9;
  const size_t offset = elementIndex * stride + memberIndex * 3;

  if (offset + 3 > buffer.size()) {
    throw std::out_of_range("SPZ buffer access out of range");
  }

  uint32_t val = static_cast<uint32_t>(buffer[offset]) | (static_cast<uint32_t>(buffer[offset + 1]) << 8) |
                 (static_cast<uint32_t>(buffer[offset + 2]) << 16);

  if (val & 0x800000) {
    val |= 0xFF000000;
  }

  return static_cast<int32_t>(val);
}

std::unique_ptr<DataTable> readSpz(const std::string& filename) {
  std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    throw std::runtime_error("cannot open file");
  }

  std::streamsize filesize = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(filesize);
  ifs.read(reinterpret_cast<char*>(buffer.data()), filesize);

  if (buffer.size() > 2 && buffer[0] == 0x1F && buffer[1] == 0x8B) {
    buffer = decompressGZIP(buffer);
  }

  const uint8_t* data = buffer.data();
  if (buffer.size() < SPZ_HEADER_SIZE) throw std::runtime_error("File too small");

  uint32_t magic;
  std::memcpy(&magic, data, 4);
  if (magic != 0x5053474E) throw std::runtime_error("Invalid SPZ magic (NGSP)");

  uint32_t version;
  std::memcpy(&version, data + 4, 4);
  uint32_t numSplats;
  std::memcpy(&numSplats, data + 8, 4);

  uint8_t shDegree = data[12];
  uint8_t fractionalBits = data[13];
  size_t harmonicsCount = HARMONICS_COMPONENT_COUNT[shDegree > 3 ? 0 : shDegree];

  size_t offset = SPZ_HEADER_SIZE;
  const uint8_t* posBase = data + offset;
  offset += numSplats * 3 * 3;
  const uint8_t* alphaBase = data + offset;
  offset += numSplats;
  const uint8_t* colorBase = data + offset;
  offset += numSplats * 3;
  const uint8_t* scaleBase = data + offset;
  offset += numSplats * 3;
  const uint8_t* rotBase = data + offset;
  offset += numSplats * (version == 3 ? 4 : 3);
  const uint8_t* shBase = data + offset;

  std::vector<Column> columns = {// Position
                                 {"x", std::vector<float>(numSplats, 0.0f)},
                                 {"y", std::vector<float>(numSplats, 0.0f)},
                                 {"z", std::vector<float>(numSplats, 0.0f)},

                                 // Scale (stored as linear in .splat, convert to log for internal use)
                                 {"scale_0", std::vector<float>(numSplats, 0.0f)},
                                 {"scale_1", std::vector<float>(numSplats, 0.0f)},
                                 {"scale_2", std::vector<float>(numSplats, 0.0f)},

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

  for (size_t i = 0; i < harmonicsCount; ++i)
    columns.push_back({"f_rest_" + std::to_string(i), std::vector<float>(numSplats, 0.0f)});

  std::vector<float*> colPtrs;
  for (auto& col : columns) colPtrs.push_back(col.asVector<float>().data());

  const float posScale = 1.0f / (1 << fractionalBits);
  for (uint32_t i = 0; i < numSplats; ++i) {
    // Position (24-bit fixed point)
    colPtrs[0][i] = getFixed24(buffer, i, 0) * posScale;
    colPtrs[1][i] = getFixed24(buffer, i, 1) * posScale;
    colPtrs[2][i] = getFixed24(buffer, i, 2) * posScale;

    // Scale
    colPtrs[3][i] = scaleBase[i * 3 + 0] / 16.0f - 10.0f;
    colPtrs[4][i] = scaleBase[i * 3 + 1] / 16.0f - 10.0f;
    colPtrs[5][i] = scaleBase[i * 3 + 2] / 16.0f - 10.0f;

    // Color & Opacity (Inverse Sigmoid)
    colPtrs[6][i] = inverseConvertColorFromSPZ(colorBase[i * 3 + 0]);
    colPtrs[7][i] = inverseConvertColorFromSPZ(colorBase[i * 3 + 1]);
    colPtrs[8][i] = inverseConvertColorFromSPZ(colorBase[i * 3 + 2]);

    float normAlpha = std::clamp(alphaBase[i] / 255.0f, 1e-6f, 1.0f - 1e-6f);
    colPtrs[9][i] = std::log(normAlpha / (1.0f - normAlpha));

    // Rotation (Quaternions)
    float q[4] = {1, 0, 0, 0};
    if (version == 2) {
      q[1] = (rotBase[i * 3 + 0] / 127.5f) - 1.0f;
      q[2] = (rotBase[i * 3 + 1] / 127.5f) - 1.0f;
      q[3] = (rotBase[i * 3 + 2] / 127.5f) - 1.0f;
      float dot = q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
      q[0] = std::sqrt(std::max(0.0f, 1.0f - dot));
    } else if (version == 3) {
      uint32_t packed;
      std::memcpy(&packed, rotBase + i * 4, 4);
      uint32_t largestIndex = packed >> 30;
      float sum_sq = 0;
      uint32_t temp = packed;
      for (int j = 3; j >= 0; --j) {
        if (static_cast<uint32_t>(j) != largestIndex) {
          uint32_t mag = temp & 511;
          float val = 0.70710678f * mag / 511.0f;
          if ((temp >> 9) & 1) val = -val;
          q[j] = val;
          sum_sq += val * val;
          temp >>= 10;
        }
      }
      q[largestIndex] = std::sqrt(std::max(0.0f, 1.0f - sum_sq));
    }
    colPtrs[10][i] = q[0];
    colPtrs[11][i] = q[1];
    colPtrs[12][i] = q[2];
    colPtrs[13][i] = q[3];

    // Spherical Harmonics
    for (size_t sh = 0; sh < harmonicsCount; ++sh) {
      size_t channel = sh % 3;
      size_t coeff = sh / 3;
      size_t colIdx = 14 + (channel * (harmonicsCount / 3) + coeff);
      uint8_t shVal = shBase[i * harmonicsCount + sh];
      colPtrs[colIdx][i] = (static_cast<float>(shVal) - 128.0f) / 128.0f;
    }
  }

  return std::make_unique<DataTable>(std::move(columns));
}

}  // namespace splat
