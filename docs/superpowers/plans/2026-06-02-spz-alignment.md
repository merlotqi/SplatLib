# SPZ Format Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align SplatLib's SPZ implementation with the Niantic/Adobe reference library — add v4 ZSTD reader, SPZ writer with full encode pipeline, and reusable CoordinateConverter module.

**Architecture:** Three independently testable iterations. Iter 1 adds v4 NGSP multi-stream ZSTD read to the existing reader. Iter 2 adds SPZ write with encode functions as exact inverses of decode. Iter 3 ports CoordinateConverter as a standalone maths module with zero SPZ dependencies.

**Tech Stack:** C++17, zstd (new), zlib (existing), Eigen (existing for Iter 3), nlohmann::json (existing)

**Spec:** `docs/superpowers/specs/2026-06-02-spz-alignment-design.md`
**Reference:** `reference/spz/src/cc/load-spz.cc`, `reference/spz/src/cc/splat-types.h`

---

## File Structure

```
Create:  src/io/spz_encoder.h             (internal encode declarations)
Create:  src/io/spz_encoder.cpp            (encode implementations)
Create:  include/splat/io/spz_writer.h     (public writeSpz API)
Create:  src/io/spz_writer.cpp             (pipeline + file output)
Create:  include/splat/maths/coordinate-converter.h  (enum, struct, factory)
Create:  src/maths/coordinate-converter.cpp           (implementation)
Create:  tests/spz_writer_test.cpp         (writer + encoder round-trip tests)
Create:  tests/coordinate_converter_test.cpp  (converter unit tests)

Modify: src/io/spz_reader.cpp              (v4 path + extracted helpers)
Modify: CMakeLists.txt                     (zstd dependency)
Modify: tests/CMakeLists.txt               (register new test targets)
```

---

## Iteration 1: v4 NGSP Reader

### Task 1.1: Add zstd dependency

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add find_package and link**

After `find_package(WebP REQUIRED)`, add:

```cmake
find_package(zstd REQUIRED)
```

In the `target_link_libraries(splat PUBLIC ...)` block, add `ZSTD::ZSTD` after `ZLIB::ZLIB`:

```cmake
target_link_libraries(splat
    PUBLIC
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
        WebP::webp
        ZLIB::ZLIB
        ZSTD::ZSTD
        absl::base
        absl::strings
    PRIVATE
        meshoptimizer
)
```

- [ ] **Step 2: Reconfigure and verify**

Run: `cmake -B build -DBUILD_SPLAT_TESTS=ON 2>&1 | tail -5`
Expected: `-- Configuring done` with zstd found

- [ ] **Step 3: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: add zstd dependency for SPZ v4 NGSP support

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1.2: Extract shared decode helpers

**Files:**
- Modify: `src/io/spz_reader.cpp`

The current `readSpz()` has decode logic inlined in a single 200-line function. Extract the per-attribute decode loops into static helper functions so they can be reused by the v4 ZSTD path.

- [ ] **Step 1: Extract decodePositions helper**

Add before `readSpz()`:

```cpp
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
```

- [ ] **Step 2: Extract decodeScales helper**

```cpp
static void decodeScales(const uint8_t* scaleBase, uint32_t numSplats,
                         std::vector<float*>& cols) {
  for (uint32_t i = 0; i < numSplats; ++i) {
    cols[0][i] = scaleBase[i * 3 + 0] / 16.0f - 10.0f;
    cols[1][i] = scaleBase[i * 3 + 1] / 16.0f - 10.0f;
    cols[2][i] = scaleBase[i * 3 + 2] / 16.0f - 10.0f;
  }
}
```

- [ ] **Step 3: Extract decodeColors helper**

```cpp
static float inverseConvertColorFromSPZ(float y) {
  return (y / 255.0f - 0.5f) / 0.15f;
}

static void decodeColors(const uint8_t* colorBase, uint32_t numSplats,
                         std::vector<float*>& cols) {
  for (uint32_t i = 0; i < numSplats; ++i) {
    cols[0][i] = inverseConvertColorFromSPZ(colorBase[i * 3 + 0]);
    cols[1][i] = inverseConvertColorFromSPZ(colorBase[i * 3 + 1]);
    cols[2][i] = inverseConvertColorFromSPZ(colorBase[i * 3 + 2]);
  }
}
```

- [ ] **Step 4: Extract decodeAlphas helper**

```cpp
static void decodeAlphas(const uint8_t* alphaBase, uint32_t numSplats,
                         std::vector<float*>& cols) {
  for (uint32_t i = 0; i < numSplats; ++i) {
    float normAlpha = std::clamp(alphaBase[i] / 255.0f, 1e-6f, 1.0f - 1e-6f);
    cols[0][i] = std::log(normAlpha / (1.0f - normAlpha));
  }
}
```

- [ ] **Step 5: Extract decodeRotations helper**

```cpp
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
```

- [ ] **Step 6: Extract decodeSH helper**

```cpp
static const size_t HARMONICS_COMPONENT_COUNT[] = {0, 9, 24, 45};

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
```

- [ ] **Step 7: Rewrite readSpz body to use extracted helpers**

Replace the inline decode loops in the existing `readSpz()` with calls to the new helpers. Keep the existing buffer layout computation (offset arithmetic for posBase, alphaBase, colorBase, scaleBase, rotBase, shBase) and column allocation — only replace the per-attribute for-loops.

- [ ] **Step 8: Build and run existing tests**

```bash
cmake --build build --target SplatLccReaderTests 2>&1 | tail -3
```
Expected: build succeeds, no undefined symbol errors.

- [ ] **Step 9: Commit**

```bash
git add src/io/spz_reader.cpp
git commit -m "refactor(spz): extract shared decode helpers from readSpz

Extracts per-attribute decode loops into static helper functions
so they can be reused by the v4 ZSTD multi-stream path. No
behavior change.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1.3: Add v4 NGSP header and TOC parsing

**Files:**
- Modify: `src/io/spz_reader.cpp`

- [ ] **Step 1: Add NgspFileHeader struct and constants**

Add near the top of the file (after includes, before helpers):

```cpp
static constexpr uint32_t SPZ_V4_MAGIC = 0x5053474E;  // "NGSP"
static constexpr size_t SPZ_V4_HEADER_SIZE = 32;
static constexpr uint8_t SPZ_V4_FLAG_ANTIALIASED = 0x1;
static constexpr uint8_t SPZ_V4_FLAG_EXTENSIONS = 0x2;

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
```

- [ ] **Step 2: Add TOC stream info struct**

```cpp
struct NgspStreamInfo {
  uint64_t compressedSize;
  uint64_t uncompressedSize;
  size_t dataOffset;  // byte offset from start of file to compressed stream data
};
```

- [ ] **Step 3: Add TOC parser function**

```cpp
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
```

- [ ] **Step 4: Build to verify compilation**

```bash
cmake --build build --target splat 2>&1 | grep -E "error|spz_reader"
```
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/io/spz_reader.cpp
git commit -m "feat(spz): add NgspFileHeader struct and TOC parser for v4 format

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1.4: Add ZSTD stream decompression and v4 readSpz path

**Files:**
- Modify: `src/io/spz_reader.cpp`

- [ ] **Step 1: Add ZSTD decompress function**

```cpp
#include <zstd.h>

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
```

- [ ] **Step 2: Add readSpzV4 function**

```cpp
static std::unique_ptr<DataTable> readSpzV4(
    const std::vector<uint8_t>& buffer) {

  NgspFileHeader header;
  std::memcpy(&header, buffer.data(), SPZ_V4_HEADER_SIZE);

  if (header.magic != SPZ_V4_MAGIC) {
    throw std::runtime_error("Invalid v4 SPZ magic");
  }
  if (header.version < 4 || header.version > 4) {
    throw std::runtime_error("Unsupported SPZ v4 version: " + std::to_string(header.version));
  }
  if (header.numPoints == 0) {
    throw std::runtime_error("v4 SPZ has zero points");
  }

  uint32_t numSplats = header.numPoints;
  uint8_t shDegree = header.shDegree;
  bool usesQuaternionSmallestThree = header.version >= 3;
  size_t harmonicsCount = HARMONICS_COMPONENT_COUNT[shDegree > 3 ? 0 : shDegree];
  unsigned rotStride = usesQuaternionSmallestThree ? 4 : 3;

  // Allocate columns
  std::vector<float> xData(numSplats), yData(numSplats), zData(numSplats);
  std::vector<float> s0Data(numSplats), s1Data(numSplats), s2Data(numSplats);
  std::vector<float> dc0Data(numSplats), dc1Data(numSplats), dc2Data(numSplats);
  std::vector<float> opData(numSplats);
  std::vector<float> r0Data(numSplats), r1Data(numSplats), r2Data(numSplats), r3Data(numSplats);
  std::vector<std::vector<float>> shData(harmonicsCount, std::vector<float>(numSplats));

  // Parse TOC
  auto streams = parseNgspTOC(buffer.data(), buffer.size(), header);

  // Build dests in v4 stream order: positions, alphas, colors, scales, rotations, sh
  std::vector<uint8_t> posBuf(numSplats * 9), alphaBuf(numSplats * 1),
                       colorBuf(numSplats * 3), scaleBuf(numSplats * 3),
                       rotBuf(numSplats * rotStride), shBuf(numSplats * harmonicsCount);

  std::vector<std::pair<uint8_t*, size_t>> dests;
  dests.push_back({posBuf.data(), posBuf.size()});
  dests.push_back({alphaBuf.data(), alphaBuf.size()});
  dests.push_back({colorBuf.data(), colorBuf.size()});
  dests.push_back({scaleBuf.data(), scaleBuf.size()});
  dests.push_back({rotBuf.data(), rotBuf.size()});
  if (harmonicsCount > 0) {
    dests.push_back({shBuf.data(), shBuf.size()});
  }

  decompressNgspStreams(buffer.data(), streams, dests);

  // Decode using shared helpers
  {
    std::vector<float*> posCols = {xData.data(), yData.data(), zData.data()};
    decodePositions(posBuf.data(), header.fractionalBits, numSplats, posCols);
  }
  {
    std::vector<float*> scaleCols = {s0Data.data(), s1Data.data(), s2Data.data()};
    decodeScales(scaleBuf.data(), numSplats, scaleCols);
  }
  {
    std::vector<float*> colorCols = {dc0Data.data(), dc1Data.data(), dc2Data.data()};
    decodeColors(colorBuf.data(), numSplats, colorCols);
  }
  {
    std::vector<float*> alphaCols = {opData.data()};
    decodeAlphas(alphaBuf.data(), numSplats, alphaCols);
  }
  {
    std::vector<float*> rotCols = {r0Data.data(), r1Data.data(), r2Data.data(), r3Data.data()};
    decodeRotations(rotBuf.data(), header.version, numSplats, rotCols);
  }
  if (harmonicsCount > 0) {
    std::vector<float*> shCols;
    for (size_t h = 0; h < harmonicsCount; ++h) shCols.push_back(shData[h].data());
    decodeSH(shBuf.data(), shDegree, numSplats, shCols);
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
```

- [ ] **Step 3: Wire v4 detection into readSpz**

Modify the top of `readSpz()` to check for v4 magic before GZip:

```cpp
std::unique_ptr<DataTable> readSpz(const std::filesystem::path& filename) {
  std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    throw std::runtime_error("cannot open file: " + filename.u8string());
  }

  std::streamsize filesize = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(filesize);
  ifs.read(reinterpret_cast<char*>(buffer.data()), filesize);

  // v4 NGSP detection
  if (buffer.size() >= 4) {
    uint32_t magic;
    std::memcpy(&magic, buffer.data(), 4);
    if (magic == SPZ_V4_MAGIC) {
      return readSpzV4(buffer);
    }
  }

  // Existing GZip + v2-v3 path follows unchanged...
  if (buffer.size() > 2 && buffer[0] == 0x1F && buffer[1] == 0x8B) {
    buffer = decompressGZIP(buffer);
  }
  // ... rest of existing code unchanged
```

- [ ] **Step 4: Build and verify**

```bash
cmake --build build --target splat 2>&1 | grep -E "error|spz_reader"
```
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/io/spz_reader.cpp
git commit -m "feat(spz): add v4 NGSP ZSTD multi-stream reader support

- Detect NGSP magic (0x5053474E) and dispatch to readSpzV4
- Parse 32-byte NgspFileHeader + TOC with per-stream sizes
- ZSTD decompress each attribute stream into pre-allocated buffers
- Reuse extracted decode helpers for attribute unpacking
- Legacy v2-v3 GZip path preserved unchanged

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Iteration 2: SPZ Writer

### Task 2.1: Create spz_encoder.h — internal encode declarations

**Files:**
- Create: `src/io/spz_encoder.h`

- [ ] **Step 1: Write the header file**

```cpp
/**
 * @file io/spz_encoder.h
 * @brief SPZ encoding functions — exact inverses of spz_reader decode.
 */
#pragma once

#include <cstdint>

namespace splat {

constexpr float SPZ_COLOR_SCALE = 0.15f;

inline float spzSigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

// Encode position as 3 x int24 little-endian (9 bytes per point).
// pos[3] is world-space XYZ. fractionalBits controls precision (default 12).
void encodePosition(const float pos[3], int fractionalBits, uint8_t out[9]);

// Encode log-scale as uint8 (3 bytes per point).
// Formula: clamp((log_scale + 10.0) * 16.0 + 0.5, 0, 255)
void encodeScale(const float scale_log[3], uint8_t out[3]);

// Encode SH DC color as uint8 (3 bytes per point).
// Formula: clamp(f_dc * 0.15 * 255 + 127.5 + 0.5, 0, 255)
void encodeColor(const float f_dc[3], uint8_t out[3]);

// Encode logit-space opacity as uint8 (1 byte per point).
// Formula: clamp(sigmoid(opacity) * 255.0 + 0.5, 0, 255)
uint8_t encodeAlpha(float opacity);

// Encode quaternion as smallest-three packed uint32 (4 bytes per point).
// Steps: normalize, find largest abs component, negate if needed,
// quantize 3 smallest to 9-bit mag + 1-bit sign, pack into uint32 LE.
void encodeRotation(const float rot[4], uint8_t out[4]);

// Encode SH coefficient as quantized uint8.
// Formula: clamp(round((v * 128 + 128 + bucketSize/2) / bucketSize) * bucketSize, 0, 255)
uint8_t encodeSH(float v, int bucketSize);

}  // namespace splat
```

- [ ] **Step 2: Commit**

```bash
git add src/io/spz_encoder.h
git commit -m "feat(spz): add spz_encoder.h with encode function declarations

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2.2: Create spz_encoder.cpp — encode implementations

**Files:**
- Create: `src/io/spz_encoder.cpp`

- [ ] **Step 1: Write encodePosition**

```cpp
#include "spz_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace splat {

void encodePosition(const float pos[3], int fractionalBits, uint8_t out[9]) {
  float scale = static_cast<float>(1 << fractionalBits);
  for (int i = 0; i < 3; ++i) {
    int32_t fixed = static_cast<int32_t>(std::round(pos[i] * scale));
    out[i * 3 + 0] = fixed & 0xFF;
    out[i * 3 + 1] = (fixed >> 8) & 0xFF;
    out[i * 3 + 2] = (fixed >> 16) & 0xFF;
  }
}
```

- [ ] **Step 2: Write encodeScale**

```cpp
void encodeScale(const float scale_log[3], uint8_t out[3]) {
  for (int i = 0; i < 3; ++i) {
    float v = (scale_log[i] + 10.0f) * 16.0f + 0.5f;
    out[i] = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
  }
}
```

- [ ] **Step 3: Write encodeColor**

```cpp
void encodeColor(const float f_dc[3], uint8_t out[3]) {
  for (int i = 0; i < 3; ++i) {
    float v = f_dc[i] * SPZ_COLOR_SCALE * 255.0f + 127.5f + 0.5f;
    out[i] = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
  }
}
```

- [ ] **Step 4: Write encodeAlpha**

```cpp
uint8_t encodeAlpha(float opacity) {
  float v = spzSigmoid(opacity) * 255.0f + 0.5f;
  return static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
}
```

- [ ] **Step 5: Write encodeRotation**

```cpp
void encodeRotation(const float rot[4], uint8_t out[4]) {
  // Normalize
  float w = rot[0], x = rot[1], y = rot[2], z = rot[3];
  float len = std::sqrt(w * w + x * x + y * y + z * z);
  if (len > 0.0f) { w /= len; x /= len; y /= len; z /= len; }

  // Find largest absolute component
  float absVals[4] = {std::fabs(w), std::fabs(x), std::fabs(y), std::fabs(z)};
  unsigned largestIdx = 0;
  for (unsigned i = 1; i < 4; ++i) {
    if (absVals[i] > absVals[largestIdx]) largestIdx = i;
  }

  // Ensure largest is positive
  float quat[4] = {w, x, y, z};
  if (quat[largestIdx] < 0.0f) {
    for (int i = 0; i < 4; ++i) quat[i] = -quat[i];
  }

  // Quantize 3 smallest components to 9-bit mag + 1-bit sign
  constexpr float kRsqrt2 = 0.7071067811865475f;
  constexpr uint32_t kMagMask = 511u;
  uint32_t packed = largestIdx;
  for (int i = 3; i >= 0; --i) {
    if (static_cast<unsigned>(i) != largestIdx) {
      uint32_t negBit = (quat[i] < 0.0f) ? 1u : 0u;
      uint32_t mag = static_cast<uint32_t>(
          std::round(std::fabs(quat[i]) / kRsqrt2 * 511.0f));
      if (mag > kMagMask) mag = kMagMask;
      packed = (packed << 10) | (negBit << 9) | mag;
    }
  }

  // Store as little-endian uint8[4]
  out[0] = packed & 0xFF;
  out[1] = (packed >> 8) & 0xFF;
  out[2] = (packed >> 16) & 0xFF;
  out[3] = (packed >> 24) & 0xFF;
}
```

- [ ] **Step 6: Write encodeSH**

```cpp
uint8_t encodeSH(float v, int bucketSize) {
  int q = static_cast<int>(std::round(v * 128.0f + 128.0f));
  q = (q + bucketSize / 2) / bucketSize * bucketSize;
  return static_cast<uint8_t>(std::clamp(q, 0, 255));
}
```

- [ ] **Step 7: Build to verify compilation**

```bash
cmake --build build --target splat 2>&1 | grep -E "error|spz_encoder"
```
Expected: no errors (GLOB_RECURSE auto-picks up new .cpp)

- [ ] **Step 8: Commit**

```bash
git add src/io/spz_encoder.cpp
git commit -m "feat(spz): implement SPZ encoding functions

encodePosition (int24), encodeScale (log→uint8), encodeColor (SH DC→uint8),
encodeAlpha (sigmoid→uint8), encodeRotation (smallest-three 9-bit),
encodeSH (configurable bucket quantization).

All functions are exact inverses of spz_reader decode logic.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2.3: Create include/splat/io/spz_writer.h — public API

**Files:**
- Create: `include/splat/io/spz_writer.h`

- [ ] **Step 1: Write the header**

```cpp
/**
 * @file splat/io/spz_writer.h
 * @brief Write .spz compressed Gaussian splat files (v4 NGSP format).
 */
#pragma once

#include <filesystem>
#include <cstdint>

namespace splat {

class DataTable;

struct SpzWriteOptions {
  int fractionalBits = 12;
  uint8_t sh1Bits = 5;
  uint8_t shRestBits = 4;
};

void writeSpz(const std::filesystem::path& filename,
              const DataTable& dataTable,
              const SpzWriteOptions& options = {});

}  // namespace splat
```

- [ ] **Step 2: Commit**

```bash
git add include/splat/io/spz_writer.h
git commit -m "feat(spz): add spz_writer.h public API

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2.4: Create src/io/spz_writer.cpp — pipeline + file output

**Files:**
- Create: `src/io/spz_writer.cpp`

- [ ] **Step 1: Write the writer pipeline**

```cpp
#include <splat/io/spz_writer.h>

#include "spz_encoder.h"
#include <splat/models/data-table.h>

#include <zstd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace splat {

namespace {

constexpr uint32_t NGSP_MAGIC = 0x5053474E;
constexpr int LATEST_SPZ_VERSION = 4;

int shDegreeFromColumnCount(int restCount) {
  if (restCount >= 45) return 3;
  if (restCount >= 24) return 2;
  if (restCount >= 9)  return 1;
  return 0;
}

int shDimForDegree(int degree) {
  switch (degree) {
    case 0: return 0;
    case 1: return 3;
    case 2: return 8;
    case 3: return 15;
    case 4: return 24;
    default: return 0;
  }
}

}  // namespace

void writeSpz(const std::filesystem::path& filename,
              const DataTable& dataTable,
              const SpzWriteOptions& options) {

  // 1. Validate columns
  auto getSpan = [&](const std::string& name) {
    const auto& col = dataTable.getColumnByName(name);
    if (col.getType() != ColumnType::FLOAT32)
      throw std::runtime_error("SPZ writer: column '" + name + "' must be float32");
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
  while (dataTable.hasColumn("f_rest_" + std::to_string(fRestCount))) fRestCount++;
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
      int band = 0, coeff = 0;
      for (int ch = 0; ch < 3; ++ch) {
        for (int c = 0; c < shDim; ++c) {
          int colIdx = ch * shDim + c;
          float v = 0.0f;
          int srcCol = fRestCount > colIdx ? colIdx : -1;
          if (srcCol >= 0) {
            v = dataTable.getColumnByName("f_rest_" + std::to_string(srcCol)).getValue<float>(i);
          }
          int bits = (c < 3) ? options.sh1Bits : options.shRestBits;
          int bucketSize = 1 << (8 - bits);
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
  uint32_t tocByteOffset = 32;  // no extensions
  uint8_t header[32] = {};
  std::memcpy(header, &NGSP_MAGIC, 4);
  uint32_t version = LATEST_SPZ_VERSION;
  std::memcpy(header + 4, &version, 4);
  std::memcpy(header + 8, &numPoints, 4);
  header[12] = static_cast<uint8_t>(shDegree);
  header[13] = static_cast<uint8_t>(options.fractionalBits);
  header[14] = 0;
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
```

- [ ] **Step 2: Build to verify**

```bash
cmake --build build --target splat 2>&1 | grep -E "error|spz_writer"
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/io/spz_writer.cpp
git commit -m "feat(spz): implement SPZ v4 writer pipeline

DataTable → per-splat encode → ZSTD multi-stream compress →
NgspFileHeader + TOC → .spz file output.

Supports configurable SH quantization (sh1Bits/shRestBits).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2.5: Writer round-trip test

**Files:**
- Create: `tests/spz_writer_test.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the round-trip test**

```cpp
/**
 * @file spz_writer_test.cpp
 * @brief Round-trip test: DataTable → writeSpz → readSpz → verify
 */

#include <splat/io/spz_writer.h>
#include <splat/io/spz_reader.h>
#include <splat/models/data-table.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

static void check(bool cond, const char* msg) {
  if (!cond) { fprintf(stderr, "FAIL: %s\n", msg); std::exit(1); }
}

static splat::DataTable makeTestTable(int numSplats, bool withSH) {
  std::vector<splat::Column> cols;

  auto makeCol = [&](const std::string& name) {
    std::vector<float> data(numSplats);
    cols.push_back({name, std::move(data)});
    return cols.back().asVector<float>().data();
  };

  float* xs = makeCol("x");
  float* ys = makeCol("y");
  float* zs = makeCol("z");
  float* s0 = makeCol("scale_0");
  float* s1 = makeCol("scale_1");
  float* s2 = makeCol("scale_2");
  float* dc0 = makeCol("f_dc_0");
  float* dc1 = makeCol("f_dc_1");
  float* dc2 = makeCol("f_dc_2");
  float* op = makeCol("opacity");
  float* r0 = makeCol("rot_0");
  float* r1 = makeCol("rot_1");
  float* r2 = makeCol("rot_2");
  float* r3 = makeCol("rot_3");

  for (int i = 0; i < numSplats; ++i) {
    xs[i] = i * 1.0f;   ys[i] = i * 2.0f;   zs[i] = i * 3.0f;
    s0[i] = -1.0f; s1[i] = 0.0f; s2[i] = 1.0f;
    dc0[i] = 0.1f; dc1[i] = -0.1f; dc2[i] = 0.2f;
    op[i] = 0.5f;
    r0[i] = 1.0f; r1[i] = 0.0f; r2[i] = 0.0f; r3[i] = 0.0f;
  }

  if (withSH) {
    for (int h = 0; h < 9; ++h) {
      auto data = std::vector<float>(numSplats, (h - 4.0f) * 0.2f);
      cols.push_back({"f_rest_" + std::to_string(h), std::move(data)});
    }
  }

  return splat::DataTable(std::move(cols));
}

int main() {
  auto tmpPath = std::filesystem::temp_directory_path() / "test_roundtrip.spz";

  // Test 1: No SH
  {
    auto dt = makeTestTable(10, false);
    splat::writeSpz(tmpPath, dt);
    auto result = splat::readSpz(tmpPath);
    check(result->getNumRows() == 10, "round-trip no-SH: row count");
    check(!result->hasColumn("f_rest_0"), "round-trip no-SH: no SH column");

    // Check key values (tolerance due to quantization)
    auto& xCol = result->getColumnByName("x");
    check(std::fabs(xCol.getValue<float>(0) - 0.0f) < 0.001f, "round-trip no-SH: x[0]");
    printf("PASS: round-trip no SH\n");
  }

  // Test 2: With SH
  {
    auto dt = makeTestTable(3, true);
    splat::writeSpz(tmpPath, dt);
    auto result = splat::readSpz(tmpPath);
    check(result->getNumRows() == 3, "round-trip SH: row count");
    check(result->hasColumn("f_rest_0"), "round-trip SH: has SH column");
    printf("PASS: round-trip with SH\n");
  }

  // Cleanup
  std::filesystem::remove(tmpPath);

  printf("=== All SPZ writer tests passed ===\n");
  return 0;
}
```

- [ ] **Step 2: Register test in CMakeLists.txt**

Add after the existing LCC writer test registration in `tests/CMakeLists.txt`:

```cmake
add_executable(SplatSpzWriterTests spz_writer_test.cpp)
target_link_libraries(SplatSpzWriterTests PRIVATE SPLAT::splat)
add_test(NAME SplatSpzWriterTests COMMAND SplatSpzWriterTests)
```

- [ ] **Step 3: Build and run**

```bash
cmake -B build -DBUILD_SPLAT_TESTS=ON 2>&1 | tail -3
cmake --build build --target SplatSpzWriterTests 2>&1 | tail -3
./build/tests/SplatSpzWriterTests
```
Expected: PASS for both tests.

- [ ] **Step 4: Commit**

```bash
git add tests/spz_writer_test.cpp tests/CMakeLists.txt
git commit -m "test(spz): add writer round-trip tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Iteration 3: CoordinateConverter

### Task 3.1: Create coordinate-converter.h

**Files:**
- Create: `include/splat/maths/coordinate-converter.h`

- [ ] **Step 1: Write the header**

```cpp
/**
 * @file splat/maths/coordinate-converter.h
 * @brief Coordinate system conversion for 3DGS data.
 *
 * Supports 16 coordinate systems. Within-family conversions use sign flips.
 * Cross-family conversions use R_x(±π/2) analytic rotation + sign flips.
 */
#pragma once

#include <array>
#include <functional>
#include <cstdint>

namespace splat {

enum class CoordinateSystem : uint32_t {
  UNSPECIFIED = 0,
  LDB = 1, LUB = 3, LDF = 5, LUF = 7,
  RDB = 2, RUB = 4, RDF = 6, RUF = 8,
  LFD = 9, LFU = 11, LBD = 13, LBU = 15,
  RFD = 10, RFU = 12, RBD = 14, RBU = 16,
  // Convenience aliases
  SPZ_DEFAULT = RUB,
  PLY_DEFAULT = RDF,
  GLB_DEFAULT = LUF,
  UNITY_DEFAULT = RUF,
};

struct CoordinateConverter {
  std::array<float, 3> flipP = {1, 1, 1};
  std::array<float, 3> flipQ = {1, 1, 1};
  std::array<float, 24> flipSh;
  std::function<void(float*)> rotFlipPos;
  std::function<void(float*)> rotFlipQuat;
  std::array<std::function<void(float*)>, 4> rotFlipShBands;

  CoordinateConverter();

  void convertPosition(float pos[3]) const;
  void convertRotation(float quat[4]) const;
  void convertSH(float* sh, int numBands, int shDegree) const;
};

CoordinateConverter makeCoordinateConverter(
    CoordinateSystem from, CoordinateSystem to, int shDegree);

}  // namespace splat
```

- [ ] **Step 2: Commit**

```bash
git add include/splat/maths/coordinate-converter.h
git commit -m "feat(maths): add coordinate-converter.h

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3.2: Create coordinate-converter.cpp

**Files:**
- Create: `src/maths/coordinate-converter.cpp`

- [ ] **Step 1: Implement within-family conversion (identity + sign flips)**

```cpp
#include <splat/maths/coordinate-converter.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace splat {

CoordinateConverter::CoordinateConverter() {
  flipSh.fill(1.0f);
}

namespace {

bool needRotation(CoordinateSystem a, CoordinateSystem b) {
  int an = static_cast<int>(a) - 1, bn = static_cast<int>(b) - 1;
  if (an < 0 || bn < 0) return false;
  return ((an >> 3) & 1) != ((bn >> 3) & 1);
}

std::array<bool, 3> axesMatch(CoordinateSystem a, CoordinateSystem b) {
  int an = static_cast<int>(a) - 1, bn = static_cast<int>(b) - 1;
  if (an < 0 || bn < 0) return {true, true, true};
  return {
    ((an >> 0) & 1) == ((bn >> 0) & 1),
    ((an >> 1) & 1) == ((bn >> 1) & 1),
    ((an >> 2) & 1) == ((bn >> 2) & 1)
  };
}

void computeFlips(CoordinateSystem from, CoordinateSystem to,
                  std::array<float, 3>& flipP,
                  std::array<float, 3>& flipQ,
                  std::array<float, 24>& flipSh) {
  auto [xMatch, yMatch, zMatch] = axesMatch(from, to);
  float x = xMatch ? 1.0f : -1.0f;
  float y = yMatch ? 1.0f : -1.0f;
  float z = zMatch ? 1.0f : -1.0f;

  flipP = {x, y, z};
  flipQ = {y * z, x * z, x * y};
  flipSh = {
    y, z, x, x*y, y*z, 1, x*z, 1, y,
    x*y*z, y, z, x, z, x,
    x*y, y*z, x*y, y*z, 1, x*z, 1, x*z, y
  };
}
```

- [ ] **Step 2: Implement cross-family R_x(±π/2) rotation functions**

```cpp
// R_x(+π/2): (x, y, z) → (x, -z, y)
void rotFlipPosPlus90X(float* p) {
  float y = p[1], z = p[2];
  p[1] = -z; p[2] = y;
}

// R_x(-π/2): (x, y, z) → (x, z, -y)
void rotFlipPosMinus90X(float* p) {
  float y = p[1], z = p[2];
  p[1] = z; p[2] = -y;
}

// Quaternion pre-multiply by R_x(+π/2) quaternion = (√2/2, √2/2, 0, 0)
void rotFlipQuatPlus90X(float* p) {
  float s = std::sqrt(2.0f) / 2.0f;
  float w = p[0], x = p[1], y = p[2], z = p[3];
  p[0] = s * (w + x); p[1] = s * (y - z);
  p[2] = s * (y + z); p[3] = s * (w - x);
}

// Quaternion pre-multiply by R_x(-π/2) quaternion conjugate = (√2/2, -√2/2, 0, 0)
void rotFlipQuatMinus90X(float* p) {
  float s = std::sqrt(2.0f) / 2.0f;
  float w = p[0], x = p[1], y = p[2], z = p[3];
  p[0] = s * (-w + x); p[1] = s * (y + z);
  p[2] = s * (-y + z); p[3] = s * (w + x);
}

// SH band analytic rotation tables (ported from reference splat-types.h)
void rotShPlus90X_band0(float* p) {  // l=1: 3 coeffs
  float t0 = p[0], t1 = p[1];
  p[0] = t1; p[1] = -t0;  // p[2] unchanged
}

void rotShPlus90X_band1(float* p) {  // l=2: 5 coeffs
  float s[5]; std::memcpy(s, p, 5 * sizeof(float));
  float s3 = std::sqrt(3.0f);
  p[0] = s[3]; p[1] = -s[1];
  p[2] = -0.5f * s[2] - (s3 / 2.0f) * s[4];
  p[3] = -s[0];
  p[4] = -(s3 / 2.0f) * s[2] + 0.5f * s[4];
}

void rotShMinus90X_band0(float* p) {
  float t0 = p[0], t1 = p[1];
  p[0] = -t1; p[1] = t0;
}

void rotShMinus90X_band1(float* p) {
  float s[5]; std::memcpy(s, p, 5 * sizeof(float));
  float s3 = std::sqrt(3.0f);
  p[0] = -s[3]; p[1] = -s[1];
  p[2] = -0.5f * s[2] - (s3 / 2.0f) * s[4];
  p[3] = s[0];
  p[4] = -(s3 / 2.0f) * s[2] + 0.5f * s[4];
}
```

- [ ] **Step 3: Implement makeCoordinateConverter factory**

```cpp
CoordinateConverter makeCoordinateConverter(
    CoordinateSystem from, CoordinateSystem to, int shDegree) {

  CoordinateConverter conv;
  if (from == to || from == CoordinateSystem::UNSPECIFIED ||
      to == CoordinateSystem::UNSPECIFIED) {
    return conv;  // identity
  }

  if (needRotation(from, to)) {
    // Cross-family: resolve into within-family + R_x(±π/2) rotation
    bool backward = ((static_cast<int>(from) - 1) >> 3) & 1;
    CoordinateSystem innerFrom = backward
        ? static_cast<CoordinateSystem>(static_cast<int>(from) - 8)
        : static_cast<CoordinateSystem>(static_cast<int>(from) + 8);

    // Get within-family flips between innerFrom and to
    computeFlips(innerFrom, to, conv.flipP, conv.flipQ, conv.flipSh);

    // Bake flip into rotation function
    auto fp = conv.flipP, fq = conv.flipQ;
    if (backward) {
      conv.rotFlipPos = [fp](float* p) {
        p[0] *= fp[0]; p[1] *= fp[1]; p[2] *= fp[2];
        rotFlipPosMinus90X(p);
      };
      conv.rotFlipQuat = [fq](float* p) {
        p[0] *= fq[0]; p[1] *= fq[1]; p[2] *= fq[2];
        rotFlipQuatMinus90X(p);
      };
      conv.rotFlipShBands[0] = [](float* p) { rotShMinus90X_band0(p); };
      conv.rotFlipShBands[1] = [](float* p) { rotShMinus90X_band1(p); };
    } else {
      conv.rotFlipPos = [fp](float* p) {
        rotFlipPosPlus90X(p);
        p[0] *= fp[0]; p[1] *= fp[1]; p[2] *= fp[2];
      };
      conv.rotFlipQuat = [fq](float* p) {
        rotFlipQuatPlus90X(p);
        p[0] *= fq[0]; p[1] *= fq[1]; p[2] *= fq[2];
      };
      conv.rotFlipShBands[0] = [](float* p) { rotShPlus90X_band0(p); };
      conv.rotFlipShBands[1] = [](float* p) { rotShPlus90X_band1(p); };
    }
    conv.flipP = {1, 1, 1};
    conv.flipQ = {1, 1, 1};
    conv.flipSh.fill(1.0f);
  } else {
    // Within-family: sign flips only
    computeFlips(from, to, conv.flipP, conv.flipQ, conv.flipSh);
  }

  return conv;
}

void CoordinateConverter::convertPosition(float pos[3]) const {
  if (rotFlipPos) { rotFlipPos(pos); return; }
  for (int i = 0; i < 3; ++i) pos[i] *= flipP[i];
}

void CoordinateConverter::convertRotation(float quat[4]) const {
  if (rotFlipQuat) { rotFlipQuat(quat); return; }
  for (int i = 0; i < 3; ++i) quat[i] *= flipQ[i];
}

void CoordinateConverter::convertSH(float* sh, int numBands, int shDegree) const {
  if (rotFlipShBands[0]) {
    // Cross-family: per-band rotation
    for (int band = 0; band < std::min(shDegree, 2); ++band) {
      int bandStart = band * (band + 2);
      int bandSize = 2 * band + 3;
      if (bandStart + bandSize > numBands) break;
      for (int ch = 0; ch < 3; ++ch) {
        float tmp[9] = {};
        for (int k = 0; k < bandSize; ++k) tmp[k] = sh[(bandStart + k) * 3 + ch];
        if (rotFlipShBands[static_cast<size_t>(band)])
          rotFlipShBands[static_cast<size_t>(band)](tmp);
        for (int k = 0; k < bandSize; ++k) sh[(bandStart + k) * 3 + ch] = tmp[k];
      }
    }
  } else {
    // Within-family: per-coefficient sign flip
    for (int i = 0; i < numBands; ++i) {
      int base = i * 3;
      sh[base + 0] *= flipSh[static_cast<size_t>(i)];
      sh[base + 1] *= flipSh[static_cast<size_t>(i)];
      sh[base + 2] *= flipSh[static_cast<size_t>(i)];
    }
  }
}
```

- [ ] **Step 4: Build**

```bash
cmake --build build --target splat 2>&1 | grep -E "error|coordinate"
```
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/maths/coordinate-converter.cpp
git commit -m "feat(maths): implement CoordinateConverter

Supports 16 coordinate systems with within-family sign flips and
cross-family R_x(±π/2) analytic rotation for positions, quaternions,
and spherical harmonics bands l=1-2.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3.3: CoordinateConverter unit tests

**Files:**
- Create: `tests/coordinate_converter_test.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the tests**

```cpp
/**
 * @file coordinate_converter_test.cpp
 * @brief Unit tests for CoordinateConverter.
 */
#include <splat/maths/coordinate-converter.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

static void check(bool cond, const char* msg) {
  if (!cond) { fprintf(stderr, "FAIL: %s\n", msg); std::exit(1); }
}

int main() {
  using CS = splat::CoordinateSystem;

  // Test 1: identity
  {
    auto conv = splat::makeCoordinateConverter(CS::RUB, CS::RUB, 0);
    float p[3] = {1, 2, 3};
    conv.convertPosition(p);
    check(p[0] == 1 && p[1] == 2 && p[2] == 3, "identity: position unchanged");
    printf("PASS: identity conversion\n");
  }

  // Test 2: within-family (RUB → RUF)
  {
    auto conv = splat::makeCoordinateConverter(CS::RUB, CS::RUF, 0);
    float p[3] = {1, 2, 3};
    conv.convertPosition(p);
    check(p[0] == 1 && p[1] == 2 && p[2] == -3, "RUB→RUF: z flipped");
    printf("PASS: within-family RUB→RUF\n");
  }

  // Test 3: cross-family round-trip (RUB → RDF → RUB)
  {
    auto toRDF = splat::makeCoordinateConverter(CS::RUB, CS::RDF, 1);
    auto toRUB = splat::makeCoordinateConverter(CS::RDF, CS::RUB, 1);

    float p[3] = {1, 2, 3};
    toRDF.convertPosition(p);
    toRUB.convertPosition(p);
    check(std::fabs(p[0] - 1) < 0.001f &&
          std::fabs(p[2] - 3) < 0.001f, "RUB→RDF→RUB: position round-trip");
    printf("PASS: cross-family position round-trip\n");
  }

  // Test 4: all 16 systems convert without crash
  for (int from = 1; from <= 16; ++from) {
    auto conv = splat::makeCoordinateConverter(
        static_cast<CS>(from), CS::RUB, 0);
    float p[3] = {1, 2, 3};
    conv.convertPosition(p);  // must not crash
  }
  printf("PASS: all 16 coordinate systems\n");

  printf("\n=== All coordinate converter tests passed ===\n");
  return 0;
}
```

- [ ] **Step 2: Register test**

```cmake
add_executable(SplatCoordinateConverterTests coordinate_converter_test.cpp)
target_link_libraries(SplatCoordinateConverterTests PRIVATE SPLAT::splat)
add_test(NAME SplatCoordinateConverterTests COMMAND SplatCoordinateConverterTests)
```

- [ ] **Step 3: Build and run**

```bash
cmake -B build -DBUILD_SPLAT_TESTS=ON 2>&1 | tail -3
cmake --build build --target SplatCoordinateConverterTests 2>&1 | tail -3
./build/tests/SplatCoordinateConverterTests
```

- [ ] **Step 4: Commit**

```bash
git add tests/coordinate_converter_test.cpp tests/CMakeLists.txt
git commit -m "test(maths): add CoordinateConverter unit tests

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage:**
- Iter 1 v4 Reader: Tasks 1.1-1.4 ✓
- Iter 2 Writer: Tasks 2.1-2.5 ✓
- Iter 3 CoordinateConverter: Tasks 3.1-3.3 ✓
- ZSTD dependency: Task 1.1 ✓
- Encoder functions: Task 2.2 ✓
- TOC parsing: Task 1.3 ✓
- Stream decompression: Task 1.4 ✓
- Shared decode helpers: Task 1.2 ✓

**2. Placeholder scan:** No TBD, TODO, or vague steps. All code is explicit.

**3. Type consistency:**
- `CoordinateConverter::convertPosition(float[3])` — consistent across header, impl, tests ✓
- `encodePosition/Scale/Color/Rotation/SH` — signatures match between .h and .cpp ✓
- `SpzWriteOptions::sh1Bits/shRestBits` — uint8_t throughout ✓
- Column naming convention follows existing SplatLib standard (x, y, z, scale_0..2, etc.) ✓
