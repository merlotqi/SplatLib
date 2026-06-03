#include <splat/io/spz_writer.h>

#include "spz_encoder.h"
#include <splat/models/data-table.h>

#include <zstd.h>

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace splat {

namespace {

constexpr uint32_t NGSP_MAGIC = 0x5053474E;
constexpr int SPZ_VERSION = 4;

int shDegreeFromColumnCount(int restCount) {
  if (restCount >= 45) return 3;
  if (restCount >= 24) return 2;
  if (restCount >= 9) return 1;
  return 0;
}

int shDimForDegree(int degree) {
  switch (degree) {
    case 0: return 0;
    case 1: return 3;
    case 2: return 8;
    case 3: return 15;
    default: return 0;
  }
}

}  // namespace

void writeSpz(const std::filesystem::path& filename,
              const DataTable& dataTable,
              const SpzWriteOptions& options) {

  // 1. Validate required columns
  auto getSpan = [&](const std::string& name) {
    const auto& col = dataTable.getColumnByName(name);
    if (col.getType() != ColumnType::FLOAT32) {
      throw std::runtime_error("SPZ writer: column '" + name + "' must be float32");
    }
    return col.asSpan<float>();
  };

  auto xs = getSpan("x"), ys = getSpan("y"), zs = getSpan("z");
  auto s0 = getSpan("scale_0"), s1 = getSpan("scale_1"), s2 = getSpan("scale_2");
  auto dc0 = getSpan("f_dc_0"), dc1 = getSpan("f_dc_1"), dc2 = getSpan("f_dc_2");
  auto op = getSpan("opacity");
  auto r0 = getSpan("rot_0"), r1 = getSpan("rot_1"), r2 = getSpan("rot_2"), r3 = getSpan("rot_3");

  uint32_t numPoints = static_cast<uint32_t>(dataTable.getNumRows());

  // 2. Determine SH degree
  int fRestCount = 0;
  while (dataTable.hasColumn("f_rest_" + std::to_string(fRestCount))) ++fRestCount;
  int shDegree = shDegreeFromColumnCount(fRestCount);
  int shDim = shDimForDegree(shDegree);

  // 3. Allocate packed buffers
  std::vector<uint8_t> posBuf(numPoints * 9);
  std::vector<uint8_t> alphaBuf(numPoints);
  std::vector<uint8_t> colorBuf(numPoints * 3);
  std::vector<uint8_t> scaleBuf(numPoints * 3);
  std::vector<uint8_t> rotBuf(numPoints * 4);
  std::vector<uint8_t> shBuf(numPoints * shDim * 3);

  // 4. Per-splat encode
  for (uint32_t i = 0; i < numPoints; ++i) {
    float pos[3] = {xs[i], ys[i], zs[i]};
    float scl[3] = {s0[i], s1[i], s2[i]};
    float fdc[3] = {dc0[i], dc1[i], dc2[i]};
    float rot[4] = {r0[i], r1[i], r2[i], r3[i]};

    encodePosition(pos, options.fractionalBits, posBuf.data() + i * 9);
    encodeScale(scl, scaleBuf.data() + i * 3);
    encodeColor(fdc, colorBuf.data() + i * 3);
    alphaBuf[i] = encodeAlpha(op[i]);
    encodeRotation(rot, rotBuf.data() + i * 4);

    if (shDegree > 0) {
      int bucketSize1 = 1 << (8 - options.sh1Bits);
      int bucketSizeRest = 1 << (8 - options.shRestBits);
      for (int ch = 0; ch < 3; ++ch) {
        for (int c = 0; c < shDim; ++c) {
          float v = 0.0f;
          int srcCol = ch * shDim + c;
          if (srcCol < fRestCount) {
            v = dataTable.getColumnByName("f_rest_" + std::to_string(srcCol)).getValue<float>(i);
          }
          int bucketSize = (c < 3) ? bucketSize1 : bucketSizeRest;
          size_t outIdx = (i * shDim + c) * 3 + ch;
          shBuf[outIdx] = encodeSH(v, bucketSize);
        }
      }
    }
  }

  // 5. ZSTD compress non-zero buffers
  struct StreamSrc { const uint8_t* data; size_t size; };
  std::vector<StreamSrc> srcs = {
    {posBuf.data(), posBuf.size()},
    {alphaBuf.data(), alphaBuf.size()},
    {colorBuf.data(), colorBuf.size()},
    {scaleBuf.data(), scaleBuf.size()},
    {rotBuf.data(), rotBuf.size()},
  };
  if (shDegree > 0) srcs.push_back({shBuf.data(), shBuf.size()});

  std::vector<std::vector<uint8_t>> chunks;
  std::vector<uint64_t> uncompressedSizes;

  for (const auto& src : srcs) {
    if (src.size == 0) continue;
    size_t bound = ZSTD_compressBound(src.size);
    chunks.emplace_back(bound);
    size_t compressedSize = ZSTD_compress(chunks.back().data(), bound, src.data, src.size, 12);
    if (ZSTD_isError(compressedSize)) {
      throw std::runtime_error("ZSTD compression failed");
    }
    chunks.back().resize(compressedSize);
    uncompressedSizes.push_back(src.size);
  }

  // 6. Build header
  uint8_t numStreams = static_cast<uint8_t>(chunks.size());
  uint32_t tocByteOffset = 32;
  uint8_t header[32] = {};
  std::memcpy(header, &NGSP_MAGIC, 4);
  uint32_t ver = SPZ_VERSION;
  std::memcpy(header + 4, &ver, 4);
  std::memcpy(header + 8, &numPoints, 4);
  header[12] = static_cast<uint8_t>(shDegree);
  header[13] = static_cast<uint8_t>(options.fractionalBits);
  header[14] = 0;  // flags
  header[15] = numStreams;
  std::memcpy(header + 16, &tocByteOffset, 4);

  // 7. Write file
  std::ofstream out(filename, std::ios::binary);
  if (!out) throw std::runtime_error("Failed to create: " + filename.u8string());

  out.write(reinterpret_cast<const char*>(header), 32);

  // TOC
  for (size_t i = 0; i < chunks.size(); ++i) {
    uint64_t cs = chunks[i].size();
    uint64_t us = uncompressedSizes[i];
    out.write(reinterpret_cast<const char*>(&cs), 8);
    out.write(reinterpret_cast<const char*>(&us), 8);
  }

  // Compressed stream data
  for (const auto& chunk : chunks) {
    out.write(reinterpret_cast<const char*>(chunk.data()), chunk.size());
  }

  if (!out.good()) throw std::runtime_error("Failed to write SPZ file");
}

}  // namespace splat
