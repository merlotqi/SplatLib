#include <splat/io/spz_reader.h>
#include <zlib.h>
#include <zstd.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace splat {

// ── Constants ──────────────────────────────────────────────────────────────

static constexpr uint32_t SPZ_V4_MAGIC = 0x5053474E;  // "NGSP"
static constexpr size_t SPZ_V4_HEADER_SIZE = 32;
static constexpr uint8_t SPZ_V4_FLAG_ANTIALIASED = 0x1;
static constexpr uint8_t SPZ_V4_FLAG_EXTENSIONS = 0x2;
static constexpr int LATEST_SPZ_VERSION = 4;
static constexpr float SPZ_COLOR_SCALE = 0.15f;
static const size_t HARMONICS_COMPONENT_COUNT[] = {0, 9, 24, 45};

// ── v4 NGSP header ────────────────────────────────────────────────────────

#pragma pack(push, 1)
struct NgspFileHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t numPoints;
  uint8_t  shDegree;
  uint8_t  fractionalBits;
  uint8_t  flags;
  uint8_t  numStreams;
  uint32_t tocByteOffset;
  uint8_t  reserved[12];
};
#pragma pack(pop)
static_assert(sizeof(NgspFileHeader) == 32, "NgspFileHeader must be 32 bytes");

struct NgspStreamInfo {
  uint64_t compressedSize;
  uint64_t uncompressedSize;
  size_t dataOffset;
};

// ── GZip decompression ────────────────────────────────────────────────────

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

// ── Shared attribute decode helpers (v2-v3 + v4) ──────────────────────────

static float inverseConvertColorFromSPZ(float y) { return (y / 255.0f - 0.5f) / SPZ_COLOR_SCALE; }

static void decodePositions(const uint8_t* posBase, int fractionalBits,
                            uint32_t numSplats, std::vector<float*>& cols) {
  const float posScale = 1.0f / (1 << fractionalBits);
  const size_t stride = 9;
  for (uint32_t i = 0; i < numSplats; ++i) {
    for (int m = 0; m < 3; ++m) {
      const size_t offset = i * stride + m * 3;
      int32_t fixed32 = static_cast<int32_t>(posBase[offset]) |
                        (static_cast<int32_t>(posBase[offset + 1]) << 8) |
                        (static_cast<int32_t>(posBase[offset + 2]) << 16);
      if (fixed32 & 0x800000) fixed32 |= 0xFF000000;
      cols[m][i] = static_cast<float>(fixed32) * posScale;
    }
  }
}

static void decodeScales(const uint8_t* scaleBase, uint32_t numSplats,
                         std::vector<float*>& cols) {
  for (uint32_t i = 0; i < numSplats; ++i) {
    cols[0][i] = scaleBase[i * 3 + 0] / 16.0f - 10.0f;
    cols[1][i] = scaleBase[i * 3 + 1] / 16.0f - 10.0f;
    cols[2][i] = scaleBase[i * 3 + 2] / 16.0f - 10.0f;
  }
}

static void decodeColors(const uint8_t* colorBase, uint32_t numSplats,
                         std::vector<float*>& cols) {
  for (uint32_t i = 0; i < numSplats; ++i) {
    cols[0][i] = inverseConvertColorFromSPZ(colorBase[i * 3 + 0]);
    cols[1][i] = inverseConvertColorFromSPZ(colorBase[i * 3 + 1]);
    cols[2][i] = inverseConvertColorFromSPZ(colorBase[i * 3 + 2]);
  }
}

static void decodeAlphas(const uint8_t* alphaBase, uint32_t numSplats,
                         std::vector<float*>& cols) {
  for (uint32_t i = 0; i < numSplats; ++i) {
    float normAlpha = std::clamp(alphaBase[i] / 255.0f, 1e-6f, 1.0f - 1e-6f);
    cols[0][i] = std::log(normAlpha / (1.0f - normAlpha));
  }
}

static void decodeRotations(const uint8_t* rotBase, uint32_t version,
                            uint32_t numSplats, std::vector<float*>& cols) {
  for (uint32_t i = 0; i < numSplats; ++i) {
    float q[4] = {1, 0, 0, 0};
    if (version == 2) {
      q[1] = (rotBase[i * 3 + 0] / 127.5f) - 1.0f;
      q[2] = (rotBase[i * 3 + 1] / 127.5f) - 1.0f;
      q[3] = (rotBase[i * 3 + 2] / 127.5f) - 1.0f;
      float dot = q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
      q[0] = std::sqrt(std::max(0.0f, 1.0f - dot));
    } else {
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
    cols[0][i] = q[0]; cols[1][i] = q[1];
    cols[2][i] = q[2]; cols[3][i] = q[3];
  }
}

static void decodeSH(const uint8_t* shBase, uint8_t shDegree,
                     uint32_t numSplats, std::vector<float*>& cols) {
  size_t harmonicsCount = HARMONICS_COMPONENT_COUNT[shDegree > 3 ? 0 : shDegree];
  for (uint32_t i = 0; i < numSplats; ++i) {
    for (size_t sh = 0; sh < harmonicsCount; ++sh) {
      size_t channel = sh % 3;
      size_t coeff = sh / 3;
      size_t colIdx = channel * (harmonicsCount / 3) + coeff;
      uint8_t shVal = shBase[i * harmonicsCount + sh];
      cols[colIdx][i] = (static_cast<float>(shVal) - 128.0f) / 128.0f;
    }
  }
}

// ── v4 NGSP TOC and ZSTD decompression ────────────────────────────────────

static std::vector<NgspStreamInfo> parseNgspTOC(
    const uint8_t* data, size_t size, const NgspFileHeader& header) {

  if (header.tocByteOffset < SPZ_V4_HEADER_SIZE) {
    throw std::runtime_error("NGSP TOC offset is before end of header");
  }
  size_t tocSize = header.numStreams * 16;
  if (header.tocByteOffset + tocSize > size) {
    throw std::runtime_error("NGSP TOC extends past end of file");
  }

  std::vector<NgspStreamInfo> streams(header.numStreams);
  size_t compressedOffset = header.tocByteOffset + tocSize;

  for (uint8_t i = 0; i < header.numStreams; ++i) {
    size_t entryOffset = header.tocByteOffset + i * 16;
    std::memcpy(&streams[i].compressedSize, data + entryOffset, 8);
    std::memcpy(&streams[i].uncompressedSize, data + entryOffset + 8, 8);
    streams[i].dataOffset = compressedOffset;
    compressedOffset += streams[i].compressedSize;
  }

  if (compressedOffset != size) {
    throw std::runtime_error("NGSP compressed data size mismatch");
  }
  return streams;
}

static void decompressNgspStreams(
    const uint8_t* data,
    const std::vector<NgspStreamInfo>& streams,
    std::vector<std::pair<uint8_t*, size_t>>& dests) {

  if (streams.size() != dests.size()) {
    throw std::runtime_error("NGSP stream count mismatch");
  }

  for (size_t i = 0; i < streams.size(); ++i) {
    if (streams[i].uncompressedSize != dests[i].second) {
      throw std::runtime_error("NGSP stream size mismatch for stream " + std::to_string(i));
    }
    size_t ret = ZSTD_decompress(
        dests[i].first, dests[i].second,
        data + streams[i].dataOffset, streams[i].compressedSize);
    if (ZSTD_isError(ret) || ret != dests[i].second) {
      throw std::runtime_error("ZSTD decompression failed for stream " + std::to_string(i));
    }
  }
}

// ── v2-v3 legacy reader (extracted) ───────────────────────────────────────

static std::unique_ptr<DataTable> readSpzLegacy(const std::vector<uint8_t>& buffer) {
  const uint8_t* data = buffer.data();
  if (buffer.size() < 16) throw std::runtime_error("File too small");

  uint32_t magic;
  std::memcpy(&magic, data, 4);
  if (magic != 0x5053474E) throw std::runtime_error("Invalid SPZ magic (NGSP)");

  uint32_t version;
  std::memcpy(&version, data + 4, 4);
  if (version < 2 || version > 3) {
    throw std::runtime_error("Unsupported legacy SPZ version: " + std::to_string(version));
  }

  uint32_t numSplats;
  std::memcpy(&numSplats, data + 8, 4);

  uint8_t shDegree = data[12];
  size_t harmonicsCount = HARMONICS_COMPONENT_COUNT[shDegree > 3 ? 0 : shDegree];

  size_t offset = 16;
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

  std::vector<Column> columns = {
    {"x", std::vector<float>(numSplats, 0.0f)},
    {"y", std::vector<float>(numSplats, 0.0f)},
    {"z", std::vector<float>(numSplats, 0.0f)},
    {"scale_0", std::vector<float>(numSplats, 0.0f)},
    {"scale_1", std::vector<float>(numSplats, 0.0f)},
    {"scale_2", std::vector<float>(numSplats, 0.0f)},
    {"f_dc_0", std::vector<float>(numSplats, 0.0f)},
    {"f_dc_1", std::vector<float>(numSplats, 0.0f)},
    {"f_dc_2", std::vector<float>(numSplats, 0.0f)},
    {"opacity", std::vector<float>(numSplats, 0.0f)},
    {"rot_0", std::vector<float>(numSplats, 0.0f)},
    {"rot_1", std::vector<float>(numSplats, 0.0f)},
    {"rot_2", std::vector<float>(numSplats, 0.0f)},
    {"rot_3", std::vector<float>(numSplats, 0.0f)},
  };

  for (size_t i = 0; i < harmonicsCount; ++i)
    columns.push_back({"f_rest_" + std::to_string(i), std::vector<float>(numSplats, 0.0f)});

  std::vector<float*> colPtrs;
  for (auto& col : columns) colPtrs.push_back(col.asVector<float>().data());

  // Decode using shared helpers
  {
    std::vector<float*> posCols = {colPtrs[0], colPtrs[1], colPtrs[2]};
    decodePositions(posBase, data[13], numSplats, posCols);
  }
  {
    std::vector<float*> scaleCols = {colPtrs[3], colPtrs[4], colPtrs[5]};
    decodeScales(scaleBase, numSplats, scaleCols);
  }
  {
    std::vector<float*> colorCols = {colPtrs[6], colPtrs[7], colPtrs[8]};
    decodeColors(colorBase, numSplats, colorCols);
  }
  {
    std::vector<float*> alphaCols = {colPtrs[9]};
    decodeAlphas(alphaBase, numSplats, alphaCols);
  }
  {
    std::vector<float*> rotCols = {colPtrs[10], colPtrs[11], colPtrs[12], colPtrs[13]};
    decodeRotations(rotBase, version, numSplats, rotCols);
  }
  if (harmonicsCount > 0) {
    std::vector<float*> shCols;
    for (size_t h = 0; h < harmonicsCount; ++h) shCols.push_back(colPtrs[14 + h]);
    decodeSH(shBase, shDegree, numSplats, shCols);
  }

  return std::make_unique<DataTable>(std::move(columns));
}

// ── v4 NGSP reader ───────────────────────────────────────────────────────

static std::unique_ptr<DataTable> readSpzV4(const std::vector<uint8_t>& buffer) {
  NgspFileHeader header;
  std::memcpy(&header, buffer.data(), SPZ_V4_HEADER_SIZE);

  if (header.magic != SPZ_V4_MAGIC) {
    throw std::runtime_error("Invalid v4 SPZ magic");
  }
  if (header.version != LATEST_SPZ_VERSION) {
    throw std::runtime_error("Unsupported SPZ v4 version: " + std::to_string(header.version));
  }
  if (header.numPoints == 0) {
    throw std::runtime_error("v4 SPZ has zero points");
  }

  uint32_t numSplats = header.numPoints;
  uint8_t shDegree = header.shDegree;
  size_t harmonicsCount = HARMONICS_COMPONENT_COUNT[shDegree > 3 ? 0 : shDegree];

  // Allocate per-column float buffers
  std::vector<float> xData(numSplats), yData(numSplats), zData(numSplats);
  std::vector<float> s0Data(numSplats), s1Data(numSplats), s2Data(numSplats);
  std::vector<float> dc0Data(numSplats), dc1Data(numSplats), dc2Data(numSplats);
  std::vector<float> opData(numSplats);
  std::vector<float> r0Data(numSplats), r1Data(numSplats), r2Data(numSplats), r3Data(numSplats);
  std::vector<std::vector<float>> shData(harmonicsCount, std::vector<float>(numSplats));

  // Parse TOC
  auto streams = parseNgspTOC(buffer.data(), buffer.size(), header);

  // Allocate raw decode buffers in v4 stream order: positions, alphas, colors, scales, rotations, sh
  std::vector<uint8_t> posBuf(numSplats * 9), alphaBuf(numSplats),
                       colorBuf(numSplats * 3), scaleBuf(numSplats * 3),
                       rotBuf(numSplats * 4), shBuf(numSplats * harmonicsCount);

  std::vector<std::pair<uint8_t*, size_t>> dests;
  dests.push_back({posBuf.data(), posBuf.size()});
  dests.push_back({alphaBuf.data(), alphaBuf.size()});
  dests.push_back({colorBuf.data(), colorBuf.size()});
  dests.push_back({scaleBuf.data(), scaleBuf.size()});
  dests.push_back({rotBuf.data(), rotBuf.size()});
  if (harmonicsCount > 0) dests.push_back({shBuf.data(), shBuf.size()});

  decompressNgspStreams(buffer.data(), streams, dests);

  // Decode using shared helpers
  {
    std::vector<float*> cols = {xData.data(), yData.data(), zData.data()};
    decodePositions(posBuf.data(), header.fractionalBits, numSplats, cols);
  }
  {
    std::vector<float*> cols = {s0Data.data(), s1Data.data(), s2Data.data()};
    decodeScales(scaleBuf.data(), numSplats, cols);
  }
  {
    std::vector<float*> cols = {dc0Data.data(), dc1Data.data(), dc2Data.data()};
    decodeColors(colorBuf.data(), numSplats, cols);
  }
  {
    std::vector<float*> cols = {opData.data()};
    decodeAlphas(alphaBuf.data(), numSplats, cols);
  }
  {
    std::vector<float*> cols = {r0Data.data(), r1Data.data(), r2Data.data(), r3Data.data()};
    decodeRotations(rotBuf.data(), header.version, numSplats, cols);
  }
  if (harmonicsCount > 0) {
    std::vector<float*> cols;
    for (size_t h = 0; h < harmonicsCount; ++h) cols.push_back(shData[h].data());
    decodeSH(shBuf.data(), shDegree, numSplats, cols);
  }

  // Build DataTable
  std::vector<Column> columns = {
    {"x", std::move(xData)}, {"y", std::move(yData)}, {"z", std::move(zData)},
    {"scale_0", std::move(s0Data)}, {"scale_1", std::move(s1Data)}, {"scale_2", std::move(s2Data)},
    {"f_dc_0", std::move(dc0Data)}, {"f_dc_1", std::move(dc1Data)}, {"f_dc_2", std::move(dc2Data)},
    {"opacity", std::move(opData)},
    {"rot_0", std::move(r0Data)}, {"rot_1", std::move(r1Data)},
    {"rot_2", std::move(r2Data)}, {"rot_3", std::move(r3Data)},
  };
  for (size_t h = 0; h < harmonicsCount; ++h) {
    columns.push_back({"f_rest_" + std::to_string(h), std::move(shData[h])});
  }
  return std::make_unique<DataTable>(std::move(columns));
}

// ── Public API ─────────────────────────────────────────────────────────────

std::unique_ptr<DataTable> readSpz(const std::filesystem::path& filename) {
  std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    throw std::runtime_error("cannot open file: " + filename.u8string());
  }

  std::streamsize filesize = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(filesize);
  ifs.read(reinterpret_cast<char*>(buffer.data()), filesize);

  // v4 NGSP detection (ZSTD multi-stream)
  if (buffer.size() >= 4) {
    uint32_t magic;
    std::memcpy(&magic, buffer.data(), 4);
    if (magic == SPZ_V4_MAGIC) {
      // Check version byte to distinguish v4 from legacy
      uint32_t version;
      std::memcpy(&version, buffer.data() + 4, 4);
      if (version >= 4) return readSpzV4(buffer);
    }
  }

  // Legacy GZip path (v2-v3)
  if (buffer.size() > 2 && buffer[0] == 0x1F && buffer[1] == 0x8B) {
    buffer = decompressGZIP(buffer);
  }

  return readSpzLegacy(buffer);
}

}  // namespace splat
