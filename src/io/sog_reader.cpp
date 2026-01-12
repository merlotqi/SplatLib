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

#include <absl/strings/ascii.h>
#include <absl/strings/match.h>
#include <splat/io/sog_reader.h>
#include <splat/models/sog.h>
#include <splat/utils/webp-codec.h>
#include <splat/utils/zip-reader.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace splat {

static std::array<std::vector<uint16_t>, 3> decodeMeans(const std::vector<uint8_t>& lo, const std::vector<uint8_t>& hi,
                                                        size_t count) {
  std::vector<uint16_t> xs(count, 0);
  for (size_t i = 0; i < count; i++) {
    const auto o = i * 4;
    xs[i] = static_cast<uint16_t>(lo[o + 0] | (hi[o + 1] << 8));
  }
  std::vector<uint16_t> ys(count, 0);
  for (size_t i = 0; i < count; i++) {
    const auto o = i * 4;
    ys[i] = static_cast<uint16_t>(lo[o + 2] | (hi[o + 3] << 8));
  }
  std::vector<uint16_t> zs(count, 0);
  for (size_t i = 0; i < count; i++) {
    const auto o = i * 4;
    zs[i] = static_cast<uint16_t>(lo[o + 4] | (hi[o + 5] << 8));
  }
  return {xs, ys, zs};
}

static float invLogTransform(float v) {
  const float a = abs(v);
  const float e = exp(a) - 1;
  return v < 0 ? -e : e;
}

static inline std::array<float, 4> unpackQuat(uint8_t px, uint8_t py, uint8_t pz, uint8_t tag) {
  const uint8_t maxComp = tag - 252;
  const float a = static_cast<float>(px) / 255.0f * 2.0f - 1.0f;
  const float b = static_cast<float>(py) / 255.0f * 2.0f - 1.0f;
  const float c = static_cast<float>(pz) / 255.0f * 2.0f - 1.0f;
  constexpr float sqrt2 = 1.41421356237f;
  std::array<float, 4> comps = {0.0f, 0.0f, 0.0f, 0.0f};
  static constexpr std::array<std::array<uint8_t, 3>, 4> idx = {{{{1, 2, 3}}, {{0, 2, 3}}, {{0, 1, 3}}, {{0, 1, 2}}}};

  const auto& indices = idx[maxComp];
  comps[indices[0]] = a / sqrt2;
  comps[indices[1]] = b / sqrt2;
  comps[indices[2]] = c / sqrt2;

  float t = 1.0f - (comps[0] * comps[0] + comps[1] * comps[1] + comps[2] * comps[2] + comps[3] * comps[3]);
  comps[maxComp] = sqrt(std::max(0.0f, t));

  return comps;
}

static inline float sigmoidInv(float y) {
  const float e = std::min(1.0f - 1e-6f, std::max(1e-6f, y));
  return log(e / (1 - e));
}

std::unique_ptr<DataTable> readSog(const std::string& file, const std::string& sourceName) {
  std::map<std::string, std::vector<uint8_t>> entries;
  const std::string lowerName = absl::AsciiStrToLower(sourceName);
  if (absl::EndsWith(lowerName, ".sog")) {
    ZipReader zr(file);
    const auto list = zr.list();
    for (const auto& e : list) {
      entries.insert({e.name, e.readData()});
    }
  }

  auto load = [&](const std::string& name) -> std::vector<uint8_t> {
    if (entries.count(name)) {
      return entries[name];
    }

    std::filesystem::path fullPath;
    if (sourceName.empty()) {
      fullPath = name;
    } else {
      fullPath = std::filesystem::path(sourceName) / name;
    }

    std::ifstream f(fullPath, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
      throw std::runtime_error("Could not open file: " + fullPath.string());
    }

    try {
      std::streamsize size = f.tellg();
      f.seekg(0, std::ios::beg);
      std::vector<uint8_t> buffer(size);
      if (f.read(reinterpret_cast<char*>(buffer.data()), size)) {
        f.close();
        return buffer;
      } else {
        f.close();
        throw std::runtime_error("Could not read file: " + fullPath.string());
      }
    } catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
    }
    return {};
  };

  // meta.json
  const auto metaBytes = load("meta.json");
  const auto meta = Meta::parseFromJson(metaBytes);
  const int count = meta.count;

  std::vector<Column> columns = {// Position
                                 {"x", std::vector<float>(count, 0.0f)},
                                 {"y", std::vector<float>(count, 0.0f)},
                                 {"z", std::vector<float>(count, 0.0f)},

                                 // Scale (stored as linear in .splat, convert to log for internal use)
                                 {"scale_0", std::vector<float>(count, 0.0f)},
                                 {"scale_1", std::vector<float>(count, 0.0f)},
                                 {"scale_2", std::vector<float>(count, 0.0f)},

                                 // Color/opacity
                                 {"f_dc_0", std::vector<float>(count, 0.0f)},  // Red
                                 {"f_dc_1", std::vector<float>(count, 0.0f)},  // Green
                                 {"f_dc_2", std::vector<float>(count, 0.0f)},  // Blue
                                 {"opacity", std::vector<float>(count, 0.0f)},

                                 // Rotation quaternion
                                 {"rot_0", std::vector<float>(count, 0.0f)},
                                 {"rot_1", std::vector<float>(count, 0.0f)},
                                 {"rot_2", std::vector<float>(count, 0.0f)},
                                 {"rot_3", std::vector<float>(count, 0.0f)}};

  // Prepare output columns
  auto& xCol = columns[0];
  auto& yCol = columns[1];
  auto& zCol = columns[2];

  auto& scale0Col = columns[3];
  auto& scale1Col = columns[4];
  auto& scale2Col = columns[5];

  auto& dc0 = columns[6];
  auto& dc1 = columns[7];
  auto& dc2 = columns[8];

  auto& opCol = columns[9];

  auto& r0 = columns[10];
  auto& r1 = columns[11];
  auto& r2 = columns[12];
  auto& r3 = columns[13];

  // means: two textures means_l and means_u
  const auto meansLoWebp = load(meta.means.files[0]);
  const auto meansHiWebp = load(meta.means.files[1]);
  const auto& [lo, width, height] = webpcodec::decodeRGBA(meansHiWebp);
  const auto& [hi, _hw, _hh] = webpcodec::decodeRGBA(meansLoWebp);
  const auto total = width * height;
  if (total < count) {
    throw std::runtime_error("SOG means texture too small for count");
  }
  const auto mins = meta.means.mins;
  const auto maxs = meta.means.maxs;
  const auto& [xs, ys, zs] = decodeMeans(lo, hi, count);

  const auto xMin = mins[0];
  const auto xScale = (maxs[0] - mins[0]) || 1;
  const auto yMin = mins[1];
  const auto yScale = (maxs[1] - mins[1]) || 1;
  const auto zMin = mins[2];
  const auto zScale = (maxs[2] - mins[2]) || 1;

  for (int i = 0; i < count; ++i) {
    const auto lx = xMin + xScale * (xs[i] / 65535.0f);
    const auto ly = yMin + yScale * (ys[i] / 65535.0f);
    const auto lz = zMin + zScale * (zs[i] / 65535.0f);
    xCol.setValue<float>(i, invLogTransform(lx));
    yCol.setValue<float>(i, invLogTransform(ly));
    zCol.setValue<float>(i, invLogTransform(lz));
  }

  // quats
  const auto quatsWebp = load(meta.quats.files[0]);
  const auto& [qr, qw, qh] = webpcodec::decodeRGBA(quatsWebp);
  if (qw * qh < count) {
    throw std::runtime_error("SOG quats texture too small for count");
  }

  for (int i = 0; i < count; ++i) {
    const auto o = i * 4;
    const auto tag = qr[o + 3];
    if (tag < 252) {
      r0.setValue(i, 0.0f);
      r1.setValue(i, 0.0f);
      r2.setValue(i, 0.0f);
      r3.setValue(i, 1.0f);
      continue;
    }
    const auto [x, y, z, wq] = unpackQuat(qr[o], qr[o + 1], qr[o + 2], tag);
    r0.setValue<float>(i, x);
    r1.setValue<float>(i, y);
    r2.setValue<float>(i, z);
    r3.setValue<float>(i, wq);
  }

  // scales: labels + codebook
  const auto scalesWebp = load(meta.scales.files[0]);
  const auto& [sl, sw, sh] = webpcodec::decodeRGBA(scalesWebp);
  if (sw * sh < count) {
    throw std::runtime_error("SOG scales texture too small for count");
  }
  const auto sCode = meta.scales.codebook;

  for (int i = 0; i < count; ++i) {
    const auto o = i * 4;
    scale0Col.setValue<float>(i, sCode[sl[o]]);
    scale1Col.setValue<float>(i, sCode[sl[o + 1]]);
    scale2Col.setValue<float>(i, sCode[sl[o + 2]]);
  }

  // colors + opacity: sh0.webp encodes 3 labels + opacity byte
  const auto sh0Webp = load(meta.sh0.files[0]);
  const auto& [c0, cw, ch] = webpcodec::decodeRGBA(sh0Webp);
  if (cw * ch < count) {
    throw std::runtime_error("SOG sh0 texture too small for count");
  }
  const auto cCode = meta.sh0.codebook;

  for (int i = 0; i < count; i++) {
    const auto o = i * 4;
    dc0.setValue<float>(i, cCode[c0[o + 0]]);
    dc1.setValue<float>(i, cCode[c0[o + 1]]);
    dc2.setValue<float>(i, cCode[c0[o + 2]]);
    opCol.setValue<float>(i, sigmoidInv(c0[o + 3] / 255.0f));
  }

  // Note: If present, SH higher bands (shN) are reconstructed into columns below.
  // Higher-order SH (optional)
  if (meta.shN.has_value()) {
    auto bands = meta.shN->bands;
    auto paletteCount = meta.shN->count;
    static std::array<int, 4> bandItems = {0, 3, 8, 15};
    const int shCoffs = bandItems[bands];
    if (shCoffs > 0) {
      const auto codebook = meta.shN->codebook;
      const auto centroidsWebp = load(meta.shN->files[0]);
      const auto labelsWebp = load(meta.shN->files[1]);

      const auto& centroidsDecoded = webpcodec::decodeRGBA(centroidsWebp);
      const auto& labelsDecoded = webpcodec::decodeRGBA(labelsWebp);

      const auto& centroidsRGBA = std::get<0>(centroidsDecoded);
      const auto& cW = std::get<1>(centroidsDecoded);
      const auto& cH = std::get<2>(centroidsDecoded);

      const auto& labelsRGBA = std::get<0>(labelsDecoded);
      const auto& _lw = std::get<1>(labelsDecoded);
      const auto& _lh = std::get<2>(labelsDecoded);

      // Prepare f_rest_i columns
      static constexpr auto baseIdx = 14;
      for (int i = 0; i < shCoffs * 3; ++i) {
        columns.push_back({"f_rest_" + std::to_string(i), std::vector<float>(count, 0.0f)});
      }

      const int stride = 4;
      auto getCentroidPixel = [&](int centroidIndex, int coeff) -> std::array<uint8_t, 3> {
        const int cx = (centroidIndex % 64) * shCoffs + coeff;
        const int cy = (int)floor(centroidIndex / 64.0f);
        if (cx >= cW || cy >= cH) return {0u, 0u, 0u};

        const auto idx = (cy * cW + cx) * stride;
        return {centroidsRGBA[idx], centroidsRGBA[idx + 1], centroidsRGBA[idx + 2]};
      };

      for (int i = 0; i < count; ++i) {
        const auto o = i * 4;
        const uint16_t label = labelsRGBA[o] | (labelsRGBA[o + 1] << 8);  // 16-bit palette index
        if (label >= paletteCount) {
          continue;
        }
        for (int j = 0; j < shCoffs; ++j) {
          const auto& [lr, lg, lb] = getCentroidPixel(label, j);
          columns[baseIdx + j + shCoffs * 0].setValue<float>(i, codebook[lr]);
          columns[baseIdx + j + shCoffs * 1].setValue<float>(i, codebook[lg]);
          columns[baseIdx + j + shCoffs * 2].setValue<float>(i, codebook[lb]);
        }
      }
    }
  }

  return std::make_unique<DataTable>(columns);
}

}  // namespace splat
