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

#include <absl/strings/ascii.h>
#include <absl/strings/match.h>
#include <omp.h>
#include <splat/utils/logger.h>
#include <splat/spatial/kmeans.h>
#include <splat/maths/maths.h>
#include <splat/models/morton-order.h>
#include <splat/models/sog.h>
#include <splat/utils/webp-codec.h>
#include <splat/writers/sog_writer.h>
#include <splat/utils/zip-writer.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <optional>

namespace fs = std::filesystem;

namespace splat {

static const std::array<std::string, 45> shNames = {"f_rest_0",  "f_rest_1",  "f_rest_2",  "f_rest_3",  "f_rest_4",

                                                    "f_rest_5",  "f_rest_6",  "f_rest_7",  "f_rest_8",  "f_rest_9",

                                                    "f_rest_10", "f_rest_11", "f_rest_12", "f_rest_13", "f_rest_14",

                                                    "f_rest_15", "f_rest_16", "f_rest_17", "f_rest_18", "f_rest_19",

                                                    "f_rest_20", "f_rest_21", "f_rest_22", "f_rest_23", "f_rest_24",

                                                    "f_rest_25", "f_rest_26", "f_rest_27", "f_rest_28", "f_rest_29",

                                                    "f_rest_30", "f_rest_31", "f_rest_32", "f_rest_33", "f_rest_34",

                                                    "f_rest_35", "f_rest_36", "f_rest_37", "f_rest_38", "f_rest_39",

                                                    "f_rest_40", "f_rest_41", "f_rest_42", "f_rest_43", "f_rest_44"};

static std::vector<std::array<float, 2>> calcMinMax(const DataTable* dataTable,
                                                    const std::vector<std::string>& columnNames,
                                                    const std::vector<uint32_t>& indices) {
  const size_t numCols = columnNames.size();

  std::vector<std::array<float, 2>> minMax(
      numCols, {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()});

  std::vector<const Column*> targetColumns;
  for (const auto& name : columnNames) {
    targetColumns.push_back(&dataTable->getColumnByName(name));
  }

  for (uint32_t idx : indices) {
    for (size_t j = 0; j < numCols; ++j) {
      float value = targetColumns[j]->getValue<float>(idx);

      auto& [currentMin, currentMax] = minMax[j];
      if (value < currentMin) currentMin = value;
      if (value > currentMax) currentMax = value;
    }
  }

  return minMax;
}

static float logTransform(float value) { return std::copysign(1.0f, value) * logf(std::abs(value) + 1.0f); }

static std::tuple<std::unique_ptr<DataTable>, std::unique_ptr<DataTable>> cluster1d(const DataTable* dataTable,
                                                                                    int iterations) {
  const auto numColumns = dataTable->getNumColumns();
  const auto numRows = dataTable->getNumRows();

  // construct 1d points from the columns of data
  std::vector<float> data(numRows * numColumns, 0.f);
  for (size_t i = 0; i < numColumns; ++i) {
    const auto& colData = dataTable->getColumn(i).asSpan<float>();
    std::copy(colData.begin(), colData.end(), data.begin() + (i * numRows));
  }

  auto src = std::make_unique<DataTable>();
  src->addColumn({"data", std::move(data)});

  auto [centroids, labels] = kmeans(src.release(), 256, iterations);

  // order centroids smallest to largest
  auto centroidsData = centroids->getColumn(0).asSpan<float>();
  std::vector<size_t> order(centroidsData.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return centroidsData[a] < centroidsData[b]; });

  // reorder centroids
  auto tmp = centroidsData;
  for (size_t i = 0; i < order.size(); i++) {
    centroidsData[i] = tmp[order[i]];
  }

  std::vector<uint32_t> invOrder(order.size());
  for (size_t i = 0; i < order.size(); i++) {
    invOrder[order[i]] = i;
  }

  // reorder labels
  for (size_t i = 0; i < labels.size(); i++) {
    labels[i] = invOrder[labels[i]];
  }

  std::vector<Column> resultColumns;
  auto names = dataTable->getColumnNames();
  for (auto& name : names) {
    resultColumns.push_back({name, std::vector<uint8_t>(numRows)});
  }
  for (size_t i = 0; i < numColumns; i++) {
    auto&& vec = resultColumns[i].asSpan<uint8_t>();
    size_t startOffset = i * numRows;
    for (size_t r = 0; r < numRows; ++r) {
      vec[r] = static_cast<uint8_t>(labels[startOffset + r]);
    }
  }

  return {std::move(centroids), std::make_unique<DataTable>(resultColumns)};
}

void writeSog(const std::string& outputFilename, DataTable* dataTable, bool bundle, int iterations) {
  std::unique_ptr<ZipWriter> zipWriter = bundle ? std::make_unique<ZipWriter>(outputFilename) : nullptr;

  // generateIndices
  std::vector<uint32_t> indices(dataTable->getNumRows());
  std::iota(indices.begin(), indices.end(), 0);
  sortMortonOrder(dataTable, absl::MakeSpan(indices));

  const size_t numRows = indices.size();
  const size_t width = ceil(sqrt(static_cast<double>(numRows)) / 4) * 4;
  const size_t height = std::ceil(static_cast<double>(numRows) / width / 4) * 4;
  const size_t channels = 4;

  // the layout function determines how the data is packed into the output texture.
  auto writeWebp = [&](const std::string& filename, const std::vector<uint8_t>& data, size_t w, size_t h) {
    std::vector<uint8_t> webp = webpCodec::encodeLosslessRGBA(data, w, h);
    if (zipWriter) {
      zipWriter->writeFile(filename, webp);
    } else {
      fs::path pathname = fs::path(outputFilename).parent_path() / filename;
      std::ofstream out(pathname, std::ios::binary);
      out.write(reinterpret_cast<const char*>(webp.data()), webp.size());
    }
  };

  auto writeTableData = [&](const std::string& filename, const DataTable* table, size_t w, size_t h) {
    std::vector<uint8_t> data(w * h * channels, 0);
    const size_t numColumns = table->getNumColumns();
    for (size_t i = 0; i < indices.size(); i++) {
      uint32_t idx = indices[i];
      data[i * channels + 0] = table->getColumn(0).getValue<uint8_t>(idx);
      data[i * channels + 1] = numColumns > 1 ? table->getColumn(1).getValue<uint8_t>(idx) : 0;
      data[i * channels + 2] = numColumns > 2 ? table->getColumn(2).getValue<uint8_t>(idx) : 0;
      data[i * channels + 3] = numColumns > 3 ? table->getColumn(3).getValue<uint8_t>(idx) : 255;
    }
    writeWebp(filename, data, w, h);
  };

  auto writeMeans = [&]() -> std::pair<std::vector<float>, std::vector<float>> {
    std::vector<uint8_t> meansL(width * height * channels);
    std::vector<uint8_t> meansU(width * height * channels);
    static std::vector<std::string> meansNames = {"x", "y", "z"};
    auto meansMinMax = calcMinMax(dataTable, meansNames, indices);
    for (auto&& v : meansMinMax) {
      v[0] = logTransform(v[0]);
      v[1] = logTransform(v[1]);
    }

    std::vector<int> meansColumnIdxs;
    for (const auto& name : meansNames) {
      meansColumnIdxs.push_back(dataTable->getColumnIndex(name));
    }
    Row row;

#pragma omp parallel for
    for (int i = 0; i < indices.size(); i++) {
      auto process = [&](const float& value, int axisIdx) -> uint16_t {
        float val = logTransform(value);
        float minV = meansMinMax[axisIdx][0];
        float maxV = meansMinMax[axisIdx][1];

        float normalized = (val - minV) / (maxV - minV);
        return static_cast<uint16_t>(std::clamp(normalized * 65535.0f, 0.0f, 65535.0f));
      };

      dataTable->getRow(indices[i], row, meansColumnIdxs);
      uint16_t x = process(row["x"], 0);
      uint16_t y = process(row["y"], 1);
      uint16_t z = process(row["z"], 2);

      meansL[i * 4 + 0] = static_cast<uint8_t>(x & 0xff);
      meansL[i * 4 + 1] = static_cast<uint8_t>(y & 0xff);
      meansL[i * 4 + 2] = static_cast<uint8_t>(z & 0xff);
      meansL[i * 4 + 3] = 0xff;

      meansU[i * 4 + 0] = static_cast<uint8_t>((x >> 8) & 0xff);
      meansU[i * 4 + 1] = static_cast<uint8_t>((y >> 8) & 0xff);
      meansU[i * 4 + 2] = static_cast<uint8_t>((z >> 8) & 0xff);
      meansU[i * 4 + 3] = 0xff;
    }

    writeWebp("means_l.webp", meansL, width, height);
    writeWebp("means_u.webp", meansU, width, height);

    std::vector<float> _mins;
    _mins.reserve(meansMinMax.size());
    std::vector<float> _maxs;
    _maxs.reserve(meansMinMax.size());

    for (const auto& [u, v] : meansMinMax) {
      _mins.push_back(u);
      _maxs.push_back(v);
    }
    return {_mins, _maxs};
  };

  auto writeQuaternions = [&]() {
    std::vector<uint8_t> quats(width * height * channels, 0);
    static std::vector<std::string> quatsNames = {"rot_0", "rot_1", "rot_2", "rot_3"};
    std::vector<int> quatsColumnIdxs;
    for (const auto& name : quatsNames) {
      quatsColumnIdxs.push_back(dataTable->getColumnIndex(name));
    }
    std::array<float, 4> q = {0.0, 0.0, 0.0, 0.0};

#pragma omp parallel for
    for (int i = 0; i < indices.size(); i++) {
      Row row;
      dataTable->getRow(indices[i], row, quatsColumnIdxs);
      q[0] = row["rot_0"];
      q[1] = row["rot_1"];
      q[2] = row["rot_2"];
      q[3] = row["rot_3"];

      const float l = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);

      // normalize
      std::for_each(q.begin(), q.end(), [&](float& v) { v /= l; });

      // find max component
      auto it = std::max_element(q.begin(), q.end(), [](float a, float b) { return std::abs(a) < std::abs(b); });
      const size_t maxComp = std::distance(q.begin(), it);

      // invert if max component is negative
      if (q[maxComp] < 0) {
        std::for_each(q.begin(), q.end(), [](float& v) { v = -v; });
      }

      // scale by sqrt(2) to fit in [-1, 1] range
      std::for_each(q.begin(), q.end(), [](float& v) { v *= M_SQRT2; });

      static const int QUAT_IDX_MAP[4][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {0, 1, 2}};

      const int* idx = QUAT_IDX_MAP[maxComp];
      quats[i * 4 + 0] = static_cast<uint8_t>((q[idx[0]] * 0.5f + 0.5f) * 255.0f + 0.5f);
      quats[i * 4 + 1] = static_cast<uint8_t>((q[idx[1]] * 0.5f + 0.5f) * 255.0f + 0.5f);
      quats[i * 4 + 2] = static_cast<uint8_t>((q[idx[2]] * 0.5f + 0.5f) * 255.0f + 0.5f);
      quats[i * 4 + 3] = static_cast<uint8_t>(252 + maxComp);
    }

    writeWebp("quats.webp", quats, width, height);
  };

  auto writeScales = [&]() {
    auto&& [centroids, labels] =
        cluster1d(dataTable->clone({"scale_0", "scale_1", "scale_2"}).release(), iterations);

    writeTableData("scales.webp", labels.release(), width, height);

    return centroids->getColumn(0).asVector<float>();
  };

  auto writeColors = [&]() {
    auto&& [centroids, labels] =
        cluster1d(dataTable->clone({"f_dc_0", "f_dc_1", "f_dc_2"}).release(), iterations);

    // generate and store sigmoid(opacity) [0..1]
    const auto& opacity = dataTable->getColumnByName("opacity").asSpan<float>();
    std::vector<uint8_t> opacityData(opacity.size());
    for (size_t i = 0; i < numRows; i++) {
      opacityData[i] = std::max(0.0f, std::min(255.0f, sigmoid(opacity[i]) * 255.0f));
    }
    labels->addColumn({"opacity", std::move(opacityData)});

    writeTableData("sh0.webp", labels.release(), width, height);
    return centroids->getColumn(0).asVector<float>();
  };

  auto writeSH = [&](int shBands) -> Meta::SHN {
    static std::array<int, 4> _ = {0, 3, 8, 15};
    const auto shCoeffs = _.at(shBands);

    std::vector<std::string> shColumnNames;
    for (int i = 0; i < shCoeffs * 3; i++) {
      shColumnNames.push_back(shNames[i]);
    }

    // create a table with just spherical harmonics data
    // NOTE: this step should also copy the rows referenced in indices, but that's a
    // lot of duplicate data when it's unneeded (which is currently never). so that
    // means k-means is clustering the full dataset, instead of the rows referenced in
    // indices.
    auto shDataTable = dataTable->clone(shColumnNames);
    int paletteSize =
        std::min(64, static_cast<int>(std::pow(2, std::floor(std::log2(indices.size() / 1024.0f))))) * 1024;

    auto&& [centroids, labels] = kmeans(shDataTable.release(), paletteSize, iterations);

    // construct a codebook for all spherical harmonic coefficients
    auto&& codebook = cluster1d(centroids.get(), iterations);

    // write centroids
    size_t numRowsCentroids = centroids->getNumRows();
    size_t ceilRows = static_cast<size_t>(std::ceil(numRowsCentroids / 64.0f));
    std::vector<uint8_t> centroidsBuf(64 * shCoeffs * ceilRows * 4, 0);
    Row centroidsRow;
    for (size_t i = 0; i < centroids->getNumRows(); i++) {
      std::get<1>(codebook)->getRow(i, centroidsRow);
      for (int j = 0; j < shCoeffs; ++j) {
        uint8_t x = static_cast<uint8_t>(centroidsRow[shColumnNames[shCoeffs * 0 + j]]);
        uint8_t y = static_cast<uint8_t>(centroidsRow[shColumnNames[shCoeffs * 1 + j]]);
        uint8_t z = static_cast<uint8_t>(centroidsRow[shColumnNames[shCoeffs * 2 + j]]);

        centroidsBuf[i * shCoeffs * 4 + j * 4 + 0] = x;
        centroidsBuf[i * shCoeffs * 4 + j * 4 + 1] = y;
        centroidsBuf[i * shCoeffs * 4 + j * 4 + 2] = z;
        centroidsBuf[i * shCoeffs * 4 + j * 4 + 3] = 0xff;
      }
    }

    writeWebp("shN_centroids.webp", centroidsBuf, 64 * shCoeffs, ceilRows);

    // rite labels
    std::vector<uint8_t> labelsBuf(width * height * channels, 0);
    for (size_t i = 0; i < indices.size(); ++i) {
      const uint32_t label = labels[indices[i]];

      labelsBuf[i * 4 + 0] = static_cast<uint8_t>(label & 0xff);
      labelsBuf[i * 4 + 1] = static_cast<uint8_t>((label >> 8) & 0xff);
      labelsBuf[i * 4 + 2] = 0;
      labelsBuf[i * 4 + 3] = 0xff;
    }
    writeWebp("shN_labels.webp", labelsBuf, width, height);

    return {paletteSize,
            shBands,
            std::get<0>(codebook)->getColumn(0).asVector<float>(),
            {"shN_centroids.webp", "shN_labels.webp"}};
  };

  // main
  int missingIdx = -1;
  for (int i = 0; i < (int)shNames.size(); ++i) {
    if (!dataTable->hasColumn(shNames[i])) {
      missingIdx = i;
      break;
    }
  }

  int shBands = 0;
  if (missingIdx == 9)
    shBands = 1;
  else if (missingIdx == 24)
    shBands = 2;
  else if (missingIdx == -1)
    shBands = 3;
  else
    shBands = 0;

  // convert and write attributes
  LOG_INFO("begin write means");
  std::pair<std::vector<float>, std::vector<float>> meansMinMax = writeMeans();
  LOG_INFO("begin write quaternions");
  writeQuaternions();

  LOG_INFO("begin write scales");
  std::vector<float> scalesCodebook = writeScales();
  LOG_INFO("begin write colors");
  std::vector<float> colorsCodebook = writeColors();
  std::optional<Meta::SHN> shN;
  if (shBands > 0) {
    LOG_INFO("begin write shBands");
    shN = writeSH(shBands);
  }

  Meta meta;
  meta.version = 2;
  meta.asset.generator = "splat-transform v2";
  meta.count = numRows;
  meta.means.mins = meansMinMax.first;
  meta.means.maxs = meansMinMax.second;
  meta.means.files = {"means_l.webp", "means_u.webp"};
  meta.scales.codebook = scalesCodebook;
  meta.scales.files = {"scales.webp"};
  meta.quats.files = {"quats.webp"};
  meta.sh0.codebook = colorsCodebook;
  meta.sh0.files = {"sh0.webp"};
  meta.shN = shN;

  if (zipWriter) {
    zipWriter->writeFile("meta.json", meta.encodeToJson());
  } else {
    std::ofstream out(fs::path(outputFilename).parent_path() / "meta.json");
    out << meta.encodeToJson();
  }
}

}  // namespace splat
