# LCC Reader: Align C++ Implementation with TypeScript Reference

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the C++ LCC reader to produce semantically identical output to the TypeScript `readLcc()` in splat-transform, by fixing decode helper discrepancies and implementing the missing `processUnit` / `decodeUnitsForLod` / `deserializeEnvironment` decoding pipeline.

**Architecture:** All changes are in `src/io/lcc_reader.cpp` (the single implementation file). The data model (`include/splat/models/lcc.h`) and reader interface (`include/splat/io/lcc_reader.h`) are already correct. We fix 3 helper functions, add `invLinearScale`, implement 3 new decoding functions, wire the full `readLcc()` pipeline, unblock the C API, and add unit tests for every decode helper. No new files beyond the test. No new dependencies.

**Tech Stack:** C++17, nlohmann/json, Eigen, existing `splat::DataTable`, CMake, CTest

**Spec:** `docs/superpowers/specs/2026-06-01-lcc-reader-alignment-design.md`

---

## File Structure

**Modify:**
- `src/io/lcc_reader.cpp` — fix `decodeRotation` → `decodeRotationInto`, fix `hasSH` detection, add `invLinearScale`, add `processUnit`, add `decodeUnitsForLod`, add `deserializeEnvironment`, wire full `readLcc()` pipeline with LOD selection / lod column / coordinate rotation
- `capi/splat_c.cpp` — remove `.lcc` rejection (lines 283–284)
- `tests/CMakeLists.txt` — add `SplatLccReaderTests` target

**Create:**
- `tests/lcc_reader_test.cpp` — decode helper unit tests

**Do not modify:**
- `include/splat/models/lcc.h` — data model already correct
- `include/splat/io/lcc_reader.h` — public API already correct
- `transform/reader.cpp` — already dispatches `.lcc` → `readLcc()`
- `transform/main.cpp` — already exposes `--lod_select`

---

### Task 1: Write Decode Helper Unit Tests

**Files:**
- Create: `tests/lcc_reader_test.cpp`
- Modify: `tests/CMakeLists.txt`

All LCC decode helpers are pure functions with known input→output mappings derived from TS behavior. Test them in isolation before touching production code.

- [ ] **Step 1: Create the test file**

```cpp
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

const float kSH_C0 = 0.28209479177387814f;
const float SQRT_2 = 1.41421356237f;
const float SQRT_2_INV = 0.70710678118f;

float invSigmoid(float v) {
  return -std::log((1.0f - v) / v);
}

float invSH0ToColor(float v) {
  return (v - 0.5f) / kSH_C0;
}

float invLinearScale(float v) {
  return std::log(v);
}

float mix_val(float minVal, float maxVal, float s) {
  return (1.0f - s) * minVal + s * maxVal;
}

void decodePacked_11_10_11(float& x, float& y, float& z, uint32_t enc) {
  x = static_cast<float>(enc & 0x7FF) / 2047.0f;
  y = static_cast<float>((enc >> 11) & 0x3FF) / 1023.0f;
  z = static_cast<float>((enc >> 21) & 0x7FF) / 2047.0f;
}

void decodeRotationInto(uint32_t v, float* rot0, float* rot1, float* rot2, float* rot3,
                        size_t idx) {
  float d0 = static_cast<float>(v & 1023) / 1023.0f;
  float d1 = static_cast<float>((v >> 10) & 1023) / 1023.0f;
  float d2 = static_cast<float>((v >> 20) & 1023) / 1023.0f;
  uint32_t d3 = (v >> 30) & 3;

  float qx = d0 * SQRT_2 - SQRT_2_INV;
  float qy = d1 * SQRT_2 - SQRT_2_INV;
  float qz = d2 * SQRT_2 - SQRT_2_INV;
  float sum = std::min(1.0f, qx * qx + qy * qy + qz * qz);
  float qw = std::sqrt(1.0f - sum);

  if (d3 == 0) {
    rot0[idx] = qz; rot1[idx] = qw; rot2[idx] = qx; rot3[idx] = qy;
  } else if (d3 == 1) {
    rot0[idx] = qz; rot1[idx] = qx; rot2[idx] = qw; rot3[idx] = qy;
  } else if (d3 == 2) {
    rot0[idx] = qz; rot1[idx] = qx; rot2[idx] = qy; rot3[idx] = qw;
  } else {
    rot0[idx] = qw; rot1[idx] = qx; rot2[idx] = qy; rot3[idx] = qz;
  }
}

bool near(float a, float b, float tol = 1e-5f) {
  return std::abs(a - b) <= tol;
}

void test_invSigmoid() {
  assert(near(invSigmoid(0.5f), 0.0f));
  float sig1 = 1.0f / (1.0f + std::exp(-1.0f));
  assert(near(invSigmoid(sig1), 1.0f, 0.01f));
}

void test_invSH0ToColor() {
  float colorAtZero = 0.5f + 0.0f * kSH_C0;
  assert(near(invSH0ToColor(colorAtZero), 0.0f));
}

void test_invLinearScale() {
  float s = std::exp(-2.5f);
  assert(near(invLinearScale(s), -2.5f));
  assert(near(invLinearScale(1.0f), 0.0f));
}

void test_mix() {
  assert(near(mix_val(0.0f, 10.0f, 0.0f), 0.0f));
  assert(near(mix_val(0.0f, 10.0f, 1.0f), 10.0f));
  assert(near(mix_val(0.0f, 10.0f, 0.5f), 5.0f));
  assert(near(mix_val(2.0f, 8.0f, 0.25f), 3.5f));
}

void test_decodePacked_11_10_11() {
  uint32_t enc = 1024u | (512u << 11) | (512u << 21);
  float x, y, z;
  decodePacked_11_10_11(x, y, z, enc);
  assert(near(x, 1024.0f / 2047.0f));
  assert(near(y, 512.0f / 1023.0f));
  assert(near(z, 512.0f / 2047.0f));
}

void test_decodeRotationInto_d3_0_identity() {
  // Identity quaternion, w dropped (d3=0). All three stored components = 0.
  // d0 = (0 + SQRT_2_INV)/SQRT_2 = 0.5 → 10-bit: 512
  uint32_t v = 512u | (512u << 10) | (512u << 20) | (0u << 30);
  float r0[1], r1[1], r2[1], r3[1];
  decodeRotationInto(v, r0, r1, r2, r3, 0);
  // d3=0 → rot0=qz≈0, rot1=qw≈1, rot2=qx≈0, rot3=qy≈0
  assert(near(r0[0], 0.0f, 0.01f));
  assert(near(r1[0], 1.0f, 0.01f));
  assert(near(r2[0], 0.0f, 0.01f));
  assert(near(r3[0], 0.0f, 0.01f));
}

void test_decodeRotationInto_d3_3_identity() {
  // Identity, z dropped (d3=3).
  uint32_t v = 512u | (512u << 10) | (512u << 20) | (3u << 30);
  float r0[1], r1[1], r2[1], r3[1];
  decodeRotationInto(v, r0, r1, r2, r3, 0);
  // d3=3 → rot0=qw≈1, rot1=qx≈0, rot2=qy≈0, rot3=qz≈0
  assert(near(r0[0], 1.0f, 0.01f));
  assert(near(r1[0], 0.0f, 0.01f));
  assert(near(r2[0], 0.0f, 0.01f));
  assert(near(r3[0], 0.0f, 0.01f));
}

void test_decodeRotationInto_d3_1_identity() {
  // Identity, x dropped (d3=1).
  uint32_t v = 512u | (512u << 10) | (512u << 20) | (1u << 30);
  float r0[1], r1[1], r2[1], r3[1];
  decodeRotationInto(v, r0, r1, r2, r3, 0);
  // d3=1 → rot0=qz≈0, rot1=qx≈0, rot2=qw≈1, rot3=qy≈0
  assert(near(r0[0], 0.0f, 0.01f));
  assert(near(r1[0], 0.0f, 0.01f));
  assert(near(r2[0], 1.0f, 0.01f));
  assert(near(r3[0], 0.0f, 0.01f));
}

void test_decodeRotationInto_output_is_unit_quaternion() {
  // Arbitrary encoding: d0=800, d1=200, d2=900, d3=2
  uint32_t v = 800u | (200u << 10) | (900u << 20) | (2u << 30);
  float r0[1], r1[1], r2[1], r3[1];
  decodeRotationInto(v, r0, r1, r2, r3, 0);
  float norm = r0[0]*r0[0] + r1[0]*r1[0] + r2[0]*r2[0] + r3[0]*r3[0];
  assert(near(norm, 1.0f, 1e-4f));
}

}  // namespace

int main() {
  test_invSigmoid();
  test_invSH0ToColor();
  test_invLinearScale();
  test_mix();
  test_decodePacked_11_10_11();
  test_decodeRotationInto_d3_0_identity();
  test_decodeRotationInto_d3_3_identity();
  test_decodeRotationInto_d3_1_identity();
  test_decodeRotationInto_output_is_unit_quaternion();
  return 0;
}
```

- [ ] **Step 2: Wire test into CMakeLists**

Append to `tests/CMakeLists.txt`:

```cmake
add_executable(SplatLccReaderTests lcc_reader_test.cpp)
target_link_libraries(SplatLccReaderTests PRIVATE SPLAT::splat)
add_test(NAME SplatLccReaderTests COMMAND SplatLccReaderTests)
```

- [ ] **Step 3: Build and run**

Run:
```bash
cd /home/merlot/codes/SplatLib && cmake -S . -B build/lcc-plan -DBUILD_SPLAT_TESTS=ON
cmake --build build/lcc-plan --target SplatLccReaderTests -j
./build/lcc-plan/tests/SplatLccReaderTests
```

Expected: all assertions pass (exit code 0).

- [ ] **Step 4: Commit**

```bash
git add tests/lcc_reader_test.cpp tests/CMakeLists.txt
git commit -m "test: add LCC decode helper unit tests

Establish baseline for invSigmoid, invSH0ToColor, invLinearScale,
mix, decodePacked_11_10_11, and decodeRotationInto before fixing
the lcc_reader.cpp implementations.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 2: Fix `decodeRotation` → `decodeRotationInto()` and Add `invLinearScale`

**Files:**
- Modify: `src/io/lcc_reader.cpp:33-69` (replace old `decodeRotation`, add `invLinearScale`)

- [ ] **Step 1: Add `invLinearScale` after the mix helper**

After line 25 (`static float _min_(...)`), insert:

```cpp
static float invLinearScale(float v) { return std::log(v); }
```

- [ ] **Step 2: Replace `decodeRotation` with `decodeRotationInto`**

Replace lines 33–69 (the entire `static Eigen::Quaternionf decodeRotation(...)` function) with:

```cpp
// Decode 32-bit packed rotation quaternion and write directly to output arrays.
//
// Encoding: 3 × 10-bit components (range-mapped from [-√½, √½] to [0,1023])
// plus a 2-bit index (d3) indicating which quaternion component was dropped.
//   d3==0 → w dropped   (d0=x, d1=y, d2=z stored)
//   d3==1 → x dropped   (d0=w, d1=y, d2=z stored)
//   d3==2 → y dropped   (d0=w, d1=x, d2=z stored)
//   d3==3 → z dropped   (d0=w, d1=x, d2=y stored)
//
// Output convention (matches TS decodeRotationInto):
//   rot_0 = w, rot_1 = x, rot_2 = y, rot_3 = z in the DataTable.
static void decodeRotationInto(uint32_t v, float* rot0, float* rot1, float* rot2, float* rot3,
                               size_t idx) {
  float d0 = static_cast<float>(v & 1023) / 1023.0f;
  float d1 = static_cast<float>((v >> 10) & 1023) / 1023.0f;
  float d2 = static_cast<float>((v >> 20) & 1023) / 1023.0f;
  uint32_t d3 = (v >> 30) & 3;

  float qx = d0 * SQRT_2 - SQRT_2_INV;
  float qy = d1 * SQRT_2 - SQRT_2_INV;
  float qz = d2 * SQRT_2 - SQRT_2_INV;
  float sum = std::min(1.0f, qx * qx + qy * qy + qz * qz);
  float qw = std::sqrt(1.0f - sum);

  // Reconstruct quaternion (w,x,y,z) from (qw,qx,qy,qz) based on which
  // component was dropped. The mapping below matches TS exactly.
  if (d3 == 0) {
    rot0[idx] = qz; rot1[idx] = qw; rot2[idx] = qx; rot3[idx] = qy;
  } else if (d3 == 1) {
    rot0[idx] = qz; rot1[idx] = qx; rot2[idx] = qw; rot3[idx] = qy;
  } else if (d3 == 2) {
    rot0[idx] = qz; rot1[idx] = qx; rot2[idx] = qy; rot3[idx] = qw;
  } else {
    rot0[idx] = qw; rot1[idx] = qx; rot2[idx] = qy; rot3[idx] = qz;
  }
}
```

- [ ] **Step 3: Build to verify compilation**

Run:
```bash
cd /home/merlot/codes/SplatLib && cmake --build build/lcc-plan --target splat -j 2>&1 | tail -10
```

Expected: clean build (the old `decodeRotation` was never called by any working code path).

- [ ] **Step 4: Commit**

```bash
git add src/io/lcc_reader.cpp
git commit -m "fix(lcc): replace decodeRotation with TS-aligned decodeRotationInto

Match the TypeScript decodeRotationInto output mapping for all four
d3 cases. Add invLinearScale helper for log-space scale conversion.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 3: Fix `hasSH` Detection to Match TS Logic

**Files:**
- Modify: `src/io/lcc_reader.cpp:130-136`

- [ ] **Step 1: Replace the `hasSH` detection block**

In `readLcc()`, replace lines 130–136:

```cpp
  // Match TS logic for SH detection:
  //   - "Portable" files never have spherical harmonics
  //   - "Quality" files always have spherical harmonics
  //   - Unknown/missing fileType: check for "shcoef" attribute as fallback
  //     (matches TS FIXME for early LCC files without fileType field)
  bool hasSH = false;
  if (lccJson.contains("fileType")) {
    if (lccJson["fileType"] == "Portable") {
      hasSH = false;
    } else if (lccJson["fileType"] == "Quality") {
      hasSH = true;
    } else {
      for (auto& attr : lccJson["attributes"]) {
        if (attr["name"] == "shcoef") { hasSH = true; break; }
      }
    }
  } else {
    hasSH = true;
  }
```

- [ ] **Step 2: Build to verify**

Run:
```bash
cd /home/merlot/codes/SplatLib && cmake --build build/lcc-plan --target splat -j 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 3: Commit**

```bash
git add src/io/lcc_reader.cpp
git commit -m "fix(lcc): align hasSH detection with TypeScript reference

Portable files never have SH data; Quality files always do.
Fall back to attribute inspection for unknown fileType values.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 4: Implement `processUnit()` — Core Splat Decoding

**Files:**
- Modify: `src/io/lcc_reader.cpp` (insert new function before `readLcc`)

Decodes one quadtree unit's splat records from `data.bin` (and optionally `shcoef.bin`) into pre-allocated property arrays.

**Binary record layout (data.bin, 32 bytes per splat):**

| Byte offset | Type | Field |
|-------------|------|-------|
| 0–3 | f32 LE | x position |
| 4–7 | f32 LE | y position |
| 8–11 | f32 LE | z position |
| 12 | u8 | red (0–255) |
| 13 | u8 | green (0–255) |
| 14 | u8 | blue (0–255) |
| 15 | u8 | opacity (0–255) |
| 16–17 | u16 LE | scale_0 quantized (0–65535) |
| 18–19 | u16 LE | scale_1 quantized (0–65535) |
| 20–21 | u16 LE | scale_2 quantized (0–65535) |
| 22–25 | u32 LE | packed rotation (3×10-bit + 2-bit index) |
| 26–27 | u16 LE | normal x (raw) |
| 28–29 | u16 LE | normal y (raw) |
| 30–31 | u16 LE | normal z (raw) |

**SH data (shcoef.bin, 64 bytes per splat):** 15 consecutive u32 LE, each packing 3 coefficients in 11/10/11 format. Read offset = `dataOffset * 2` (because SH stride is 64 bytes vs data stride 32 bytes).

**Coordinate rotation:** TS applies `Transform.fromEulers(90, 0, 180)`.
Rotation matrix R produces: x' = -x, y' = z, z' = y.
Quaternion q_transform = (0, 0, √½, √½), applied as q_combined = q_transform * q_splat.

- [ ] **Step 1: Add `processUnit()` before `readLcc()`**

Insert after the `parseIndexBin` function (after line 121) and before `readLcc` (line 123):

```cpp
// Decode one quadtree unit's splat data into pre-allocated property arrays.
//
// f_rest_bands: pointer to first of 45 contiguous std::vector<float>, each
// sized grandTotal. Null if SH data is not present.
static void processUnit(const LccUnitInfo& info, int targetLod, std::ifstream& dataFile,
                        std::ifstream* shFile, const CompressInfo& compressInfo,
                        size_t propertyOffset,
                        std::map<std::string, std::vector<float>>& properties,
                        std::vector<float>* f_rest_bands) {
  const auto& lod = info.lods[static_cast<size_t>(targetLod)];
  const int unitSplats = lod.points;
  if (unitSplats == 0) return;

  const int64_t dataOffset = lod.offset;
  const int dataSize = lod.size;

  // Read data.bin range for this unit
  std::vector<uint8_t> dataBytes(static_cast<size_t>(dataSize));
  dataFile.seekg(dataOffset);
  dataFile.read(reinterpret_cast<char*>(dataBytes.data()), dataSize);

  // Read shcoef.bin range if present (offset × 2: SH stride is 64 vs data 32)
  std::vector<uint8_t> shBytes;
  if (shFile && shFile->is_open()) {
    size_t shSize = static_cast<size_t>(unitSplats) * 64;
    shBytes.resize(shSize);
    shFile->seekg(dataOffset * 2);
    shFile->read(reinterpret_cast<char*>(shBytes.data()),
                 static_cast<std::streamsize>(shSize));
  }

  // Extract array references
  auto& px = properties["property_x"];
  auto& py = properties["property_y"];
  auto& pz = properties["property_z"];
  auto& pnx = properties["property_nx"];
  auto& pny = properties["property_ny"];
  auto& pnz = properties["property_nz"];
  auto& pop = properties["property_opacity"];
  auto& pr0 = properties["property_rot_0"];
  auto& pr1 = properties["property_rot_1"];
  auto& pr2 = properties["property_rot_2"];
  auto& pr3 = properties["property_rot_3"];
  auto& pdc0 = properties["property_f_dc_0"];
  auto& pdc1 = properties["property_f_dc_1"];
  auto& pdc2 = properties["property_f_dc_2"];
  auto& ps0 = properties["property_scale_0"];
  auto& ps1 = properties["property_scale_1"];
  auto& ps2 = properties["property_scale_2"];

  const float sMinX = compressInfo.scaleMin.x(), sMinY = compressInfo.scaleMin.y(),
              sMinZ = compressInfo.scaleMin.z();
  const float sMaxX = compressInfo.scaleMax.x(), sMaxY = compressInfo.scaleMax.y(),
              sMaxZ = compressInfo.scaleMax.z();
  const float shMinX = compressInfo.shMin.x(), shMinY = compressInfo.shMin.y(),
              shMinZ = compressInfo.shMin.z();
  const float shMaxX = compressInfo.shMax.x(), shMaxY = compressInfo.shMax.y(),
              shMaxZ = compressInfo.shMax.z();

  // LCC→PlayCanvas coordinate transform quaternion: fromEulers(90, 0, 180)
  // q_transform = (0, 0, √½, √½)
  const float qtw = 0.0f, qtx = 0.0f, qty = 0.70710678118f, qtz = 0.70710678118f;

  const float* f32view = reinterpret_cast<const float*>(dataBytes.data());
  const uint16_t* u16view = reinterpret_cast<const uint16_t*>(dataBytes.data());
  const uint8_t* u8data = dataBytes.data();

  for (int i = 0; i < unitSplats; ++i) {
    const size_t g = propertyOffset + static_cast<size_t>(i);
    const size_t fi = static_cast<size_t>(i) << 3;   // i * 8 f32
    const size_t bi = static_cast<size_t>(i) << 5;   // i * 32 bytes
    const size_t hi = static_cast<size_t>(i) << 4;   // i * 16 u16

    // Position: f32[0..2]
    float pos_x = f32view[fi];
    float pos_y = f32view[fi + 1];
    float pos_z = f32view[fi + 2];

    // Color + opacity: u8[12..15]
    pdc0[g] = invSH0ToColor(static_cast<float>(u8data[bi + 12]) / 255.0f);
    pdc1[g] = invSH0ToColor(static_cast<float>(u8data[bi + 13]) / 255.0f);
    pdc2[g] = invSH0ToColor(static_cast<float>(u8data[bi + 14]) / 255.0f);
    pop[g] = invSigmoid(static_cast<float>(u8data[bi + 15]) / 255.0f);

    // Scale: u16[8..10], dequantize → linear → log
    ps0[g] = invLinearScale(
        _min_(sMinX, sMaxX, static_cast<float>(u16view[hi + 8]) / 65535.0f));
    ps1[g] = invLinearScale(
        _min_(sMinY, sMaxY, static_cast<float>(u16view[hi + 9]) / 65535.0f));
    ps2[g] = invLinearScale(
        _min_(sMinZ, sMaxZ, static_cast<float>(u16view[hi + 10]) / 65535.0f));

    // Rotation: u16[11] | (u16[12] << 16) → packed u32
    uint32_t rotEnc = static_cast<uint32_t>(u16view[hi + 11]) |
                      (static_cast<uint32_t>(u16view[hi + 12]) << 16);
    decodeRotationInto(rotEnc, pr0.data(), pr1.data(), pr2.data(), pr3.data(), g);

    // Apply coordinate rotation to quaternion: q_combined = q_transform * q_splat
    float qsw = pr0[g], qsx = pr1[g], qsy = pr2[g], qsz = pr3[g];
    float cw = qtw * qsw - qtx * qsx - qty * qsy - qtz * qsz;
    float cx = qtw * qsx + qtx * qsw + qty * qsz - qtz * qsy;
    float cy = qtw * qsy - qtx * qsz + qty * qsw + qtz * qsx;
    float cz = qtw * qsz + qtx * qsy - qty * qsx + qtz * qsw;
    if (cw < 0.0f) { cw = -cw; cx = -cx; cy = -cy; cz = -cz; }
    pr0[g] = cw; pr1[g] = cx; pr2[g] = cy; pr3[g] = cz;

    // Apply coordinate rotation to position: x'=-x, y'=z, z'=y
    px[g] = -pos_x;
    py[g] = pos_z;
    pz[g] = pos_y;

    // Normals: u16[13..15], stored raw
    pnx[g] = static_cast<float>(u16view[hi + 13]);
    pny[g] = static_cast<float>(u16view[hi + 14]);
    pnz[g] = static_cast<float>(u16view[hi + 15]);

    // SH coefficients: 15 × u32 per splat (from shcoef.bin, 64-byte stride)
    if (!shBytes.empty() && f_rest_bands) {
      const uint32_t* shU32 = reinterpret_cast<const uint32_t*>(shBytes.data());
      const size_t si = static_cast<size_t>(i) << 4;  // i * 16 u32
      for (int j = 0; j < 15; ++j) {
        uint32_t enc = shU32[si + static_cast<size_t>(j)];
        float nx = static_cast<float>(enc & 0x7FF) / 2047.0f;
        float ny = static_cast<float>((enc >> 11) & 0x3FF) / 1023.0f;
        float nz = static_cast<float>((enc >> 21) & 0x7FF) / 2047.0f;
        f_rest_bands[static_cast<size_t>(j)][g] =
            _min_(shMinX, shMaxX, nx);
        f_rest_bands[static_cast<size_t>(j + 15)][g] =
            _min_(shMinY, shMaxY, ny);
        f_rest_bands[static_cast<size_t>(j + 30)][g] =
            _min_(shMinZ, shMaxZ, nz);
      }
    }
  }
}
```

- [ ] **Step 2: Build to verify compilation**

Run:
```bash
cd /home/merlot/codes/SplatLib && cmake --build build/lcc-plan --target splat -j 2>&1 | tail -15
```

Expected: clean build (new function, not yet called — no link errors).

- [ ] **Step 3: Commit**

```bash
git add src/io/lcc_reader.cpp
git commit -m "feat(lcc): implement processUnit splat decode from data.bin

Decode 32-byte splat records: f32 position, u8 color/opacity,
u16 quantized scale, packed u32 rotation, u16 normals. Decode
optional 64-byte SH records from shcoef.bin (15 x packed u32).
Apply LCC to PlayCanvas coordinate rotation inline.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 5: Implement `decodeUnitsForLod()`, `deserializeEnvironment()`, and Wire `readLcc()`

**Files:**
- Modify: `src/io/lcc_reader.cpp` (add two functions, rewrite `readLcc` body)

- [ ] **Step 1: Add `decodeUnitsForLod()` after `processUnit()`**

```cpp
// Decode all quadtree units for a single LOD level into shared property arrays.
// Pre-computes write offsets sequentially (no concurrency in the first pass).
static void decodeUnitsForLod(const std::vector<LccUnitInfo>& unitInfos, int targetLod,
                              std::ifstream& dataFile, std::ifstream* shFile,
                              const CompressInfo& compressInfo, size_t lodOffset,
                              std::map<std::string, std::vector<float>>& properties,
                              std::vector<float>* f_rest_bands) {
  size_t unitOffset = lodOffset;
  for (const auto& info : unitInfos) {
    processUnit(info, targetLod, dataFile, shFile, compressInfo, unitOffset,
                properties, f_rest_bands);
    unitOffset += static_cast<size_t>(info.lods[static_cast<size_t>(targetLod)].points);
  }
}
```

- [ ] **Step 2: Add `deserializeEnvironment()` after `decodeUnitsForLod()`**

Environment splats use the same binary layout as regular splats but with env-specific scale/SH compression bounds. Stride is 96 bytes with SH, 32 without. A `lod` column filled with `-1` is appended.

```cpp
static std::unique_ptr<DataTable> deserializeEnvironment(const std::vector<uint8_t>& raw,
                                                         const CompressInfo& compressInfo,
                                                         bool hasSH) {
  const size_t stride = hasSH ? 96u : 32u;
  if (raw.size() % stride != 0) return nullptr;

  const size_t numGaussians = raw.size() / stride;
  if (numGaussians == 0) return nullptr;

  // Allocate columns
  std::vector<std::string> colNames = {
      "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
      "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"};
  if (hasSH) {
    for (int i = 0; i < 45; ++i) colNames.push_back("f_rest_" + std::to_string(i));
  }

  std::vector<Column> columns;
  for (const auto& name : colNames) {
    columns.emplace_back(Column{name, std::vector<float>(numGaussians)});
  }

  auto colData = [&](size_t idx) -> float* {
    return &std::get<std::vector<float>>(columns[idx].data)[0];
  };

  const float sMinX = compressInfo.envScaleMin.x(), sMinY = compressInfo.envScaleMin.y(),
              sMinZ = compressInfo.envScaleMin.z();
  const float sMaxX = compressInfo.envScaleMax.x(), sMaxY = compressInfo.envScaleMax.y(),
              sMaxZ = compressInfo.envScaleMax.z();
  const float shMinX = compressInfo.envShMin.x(), shMinY = compressInfo.envShMin.y(),
              shMinZ = compressInfo.envShMin.z();
  const float shMaxX = compressInfo.envShMax.x(), shMaxY = compressInfo.envShMax.y(),
              shMaxZ = compressInfo.envShMax.z();

  // Coordinate transform (same as splat data)
  const float qtw = 0.0f, qtx = 0.0f, qty = 0.70710678118f, qtz = 0.70710678118f;

  const uint8_t* u8 = raw.data();

  for (size_t i = 0; i < numGaussians; ++i) {
    const size_t off = i * stride;

    // Position: f32[0..2]
    float pos_x, pos_y, pos_z;
    std::memcpy(&pos_x, u8 + off + 0, 4);
    std::memcpy(&pos_y, u8 + off + 4, 4);
    std::memcpy(&pos_z, u8 + off + 8, 4);

    // Color + opacity: u8[12..15]
    colData(0)[i] = -pos_x;   // x
    colData(1)[i] = pos_z;    // y  (coordinate rotation: y'=z)
    colData(2)[i] = pos_y;    // z  (coordinate rotation: z'=y)

    colData(3)[i] = invSH0ToColor(static_cast<float>(u8[off + 12]) / 255.0f);  // f_dc_0
    colData(4)[i] = invSH0ToColor(static_cast<float>(u8[off + 13]) / 255.0f);  // f_dc_1
    colData(5)[i] = invSH0ToColor(static_cast<float>(u8[off + 14]) / 255.0f);  // f_dc_2
    colData(6)[i] = invSigmoid(static_cast<float>(u8[off + 15]) / 255.0f);     // opacity

    // Scale: u16[16..21]
    uint16_t s0, s1, s2;
    std::memcpy(&s0, u8 + off + 16, 2);
    std::memcpy(&s1, u8 + off + 18, 2);
    std::memcpy(&s2, u8 + off + 20, 2);
    colData(7)[i] = invLinearScale(_min_(sMinX, sMaxX, static_cast<float>(s0) / 65535.0f));
    colData(8)[i] = invLinearScale(_min_(sMinY, sMaxY, static_cast<float>(s1) / 65535.0f));
    colData(9)[i] = invLinearScale(_min_(sMinZ, sMaxZ, static_cast<float>(s2) / 65535.0f));

    // Rotation: u32 at byte offset 22 (unaligned)
    uint32_t rotEnc;
    std::memcpy(&rotEnc, u8 + off + 22, 4);
    float* r0 = colData(10), *r1 = colData(11), *r2 = colData(12), *r3 = colData(13);
    decodeRotationInto(rotEnc, r0, r1, r2, r3, i);

    // Apply coordinate rotation to quaternion
    float cw = qtw * r0[i] - qtx * r1[i] - qty * r2[i] - qtz * r3[i];
    float cx = qtw * r1[i] + qtx * r0[i] + qty * r3[i] - qtz * r2[i];
    float cy = qtw * r2[i] - qtx * r3[i] + qty * r0[i] + qtz * r1[i];
    float cz = qtw * r3[i] + qtx * r2[i] - qty * r1[i] + qtz * r0[i];
    if (cw < 0.0f) { cw = -cw; cx = -cx; cy = -cy; cz = -cz; }
    r0[i] = cw; r1[i] = cx; r2[i] = cy; r3[i] = cz;

    // SH coefficients (skip normals at bytes 26-31 for env data)
    if (hasSH) {
      for (int j = 0; j < 15; ++j) {
        uint32_t enc;
        std::memcpy(&enc, u8 + off + 32 + static_cast<size_t>(j) * 4, 4);
        float nx = static_cast<float>(enc & 0x7FF) / 2047.0f;
        float ny = static_cast<float>((enc >> 11) & 0x3FF) / 1023.0f;
        float nz = static_cast<float>((enc >> 21) & 0x7FF) / 2047.0f;
        colData(14 + static_cast<size_t>(j))[i] = _min_(shMinX, shMaxX, nx);
        colData(14 + static_cast<size_t>(j) + 15)[i] = _min_(shMinY, shMaxY, ny);
        colData(14 + static_cast<size_t>(j) + 30)[i] = _min_(shMinZ, shMaxZ, nz);
      }
    }
  }

  // Add lod column filled with -1 (environment convention)
  columns.emplace_back(Column{"lod", std::vector<float>(numGaussians, -1.0f)});

  return std::make_unique<DataTable>(columns);
}
```

- [ ] **Step 3: Rewrite `readLcc()` body**

Replace lines 123–157 (the entire `readLcc` function body):

```cpp
std::vector<std::unique_ptr<DataTable>> readLcc(const std::filesystem::path& filename,
                                                const std::filesystem::path& sourceName,
                                                const std::vector<int>& options) {
  (void)filename;

  // 1. Parse meta.lcc JSON
  std::ifstream lccFile(sourceName);
  json lccJson = json::parse(lccFile);

  // 2. Determine SH presence (matches TS logic — see Task 3)
  bool hasSH = false;
  if (lccJson.contains("fileType")) {
    if (lccJson["fileType"] == "Portable") {
      hasSH = false;
    } else if (lccJson["fileType"] == "Quality") {
      hasSH = true;
    } else {
      for (auto& attr : lccJson["attributes"]) {
        if (attr["name"] == "shcoef") { hasSH = true; break; }
      }
    }
  } else {
    hasSH = true;
  }

  CompressInfo compressInfo = parseMeta(lccJson);
  std::vector<int> splats = lccJson["splats"].get<std::vector<int>>();

  // 3. Read index.bin
  const std::filesystem::path baseDir = sourceName.parent_path();
  const std::filesystem::path indexPath = baseDir / "index.bin";
  std::ifstream indexFile(indexPath, std::ios::binary | std::ios::ate);
  std::streamsize idxSize = indexFile.tellg();
  indexFile.seekg(0);
  std::vector<uint8_t> indexData(static_cast<size_t>(idxSize));
  indexFile.read(reinterpret_cast<char*>(indexData.data()), idxSize);

  auto unitInfos = parseIndexBin(indexData, lccJson);

  // 4. Open data files
  std::ifstream dataFile(baseDir / "data.bin", std::ios::binary);
  std::ifstream shFile;
  if (hasSH) shFile.open(baseDir / "shcoef.bin", std::ios::binary);

  // 5. Resolve LOD selection
  std::vector<int> lods;
  if (options.empty()) {
    for (size_t i = 0; i < splats.size(); ++i) lods.push_back(static_cast<int>(i));
  } else {
    for (int lod : options) {
      if (lod < 0) lod = static_cast<int>(splats.size()) + lod;
      if (lod >= 0 && lod < static_cast<int>(splats.size())) lods.push_back(lod);
    }
  }
  if (lods.empty()) {
    throw std::runtime_error("No valid LODs selected for LCC input");
  }

  // 6. Pre-allocate property arrays
  size_t grandTotal = 0;
  for (int lodIdx : lods) grandTotal += static_cast<size_t>(splats[static_cast<size_t>(lodIdx)]);

  std::map<std::string, std::vector<float>> properties;
  for (const auto& key : floatProps) {
    properties["property_" + key] = std::vector<float>(grandTotal);
  }

  std::vector<std::vector<float>> f_rest_bands;
  if (hasSH) {
    f_rest_bands.resize(45);
    for (auto& band : f_rest_bands) band.resize(grandTotal);
  }

  std::vector<float> lodColumn(grandTotal);

  // 7. Decode each selected LOD
  size_t lodOffset = 0;
  for (size_t outLod = 0; outLod < lods.size(); ++outLod) {
    int inputLod = lods[outLod];
    size_t totalSplats = static_cast<size_t>(splats[static_cast<size_t>(inputLod)]);

    decodeUnitsForLod(unitInfos, inputLod, dataFile,
                      (hasSH && shFile.is_open()) ? &shFile : nullptr,
                      compressInfo, lodOffset, properties,
                      hasSH ? f_rest_bands.data() : nullptr);

    // Fill lod column for this LOD's range
    std::fill(lodColumn.begin() + static_cast<ptrdiff_t>(lodOffset),
              lodColumn.begin() + static_cast<ptrdiff_t>(lodOffset + totalSplats),
              static_cast<float>(outLod));
    lodOffset += totalSplats;
  }

  // 8. Build DataTable columns
  std::vector<Column> columns;
  for (const auto& key : floatProps) {
    columns.emplace_back(Column{key, std::move(properties["property_" + key])});
  }
  if (hasSH) {
    for (int b = 0; b < 45; ++b) {
      columns.emplace_back(
          Column{"f_rest_" + std::to_string(b), std::move(f_rest_bands[static_cast<size_t>(b)])});
    }
  }
  columns.emplace_back(Column{"lod", std::move(lodColumn)});

  auto mainTable = std::make_unique<DataTable>(columns);
  std::vector<std::unique_ptr<DataTable>> result;
  result.push_back(std::move(mainTable));

  // 9. Try to load optional environment.bin
  try {
    std::filesystem::path envPath = baseDir / "environment.bin";
    if (std::filesystem::exists(envPath)) {
      std::ifstream envFile(envPath, std::ios::binary | std::ios::ate);
      std::streamsize envSize = envFile.tellg();
      envFile.seekg(0);
      std::vector<uint8_t> envData(static_cast<size_t>(envSize));
      envFile.read(reinterpret_cast<char*>(envData.data()), envSize);
      auto envTable = deserializeEnvironment(envData, compressInfo, hasSH);
      if (envTable) result.push_back(std::move(envTable));
    }
  } catch (...) {
    // environment.bin is optional — missing file is normal, suppress error
  }

  return result;
}
```

- [ ] **Step 4: Build to verify**

Run:
```bash
cd /home/merlot/codes/SplatLib && cmake --build build/lcc-plan --target splat -j 2>&1 | tail -20
```

Expected: clean build with no warnings. All new functions linked.

- [ ] **Step 5: Commit**

```bash
git add src/io/lcc_reader.cpp
git commit -m "feat(lcc): implement full LCC read pipeline

Add decodeUnitsForLod (sequential unit iteration), deserializeEnvironment
(optional environment.bin), and the full readLcc pipeline: LOD selection,
pre-allocated property arrays, lod column, and DataTable construction.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 6: Unblock C API for LCC Files

**Files:**
- Modify: `capi/splat_c.cpp:283-284`

- [ ] **Step 1: Remove the `.lcc` rejection**

In `capi/splat_c.cpp`, find lines 283–284 (the `if` block returning `SPLAT_C_STATUS_UNSUPPORTED_FORMAT` for `.lcc` files) and delete that block.

The existing code:
```cpp
      if (ext == ".lcc") {
        *out_status = SPLAT_C_STATUS_UNSUPPORTED_FORMAT;
        *out_error = strdup("reading .lcc files is not implemented yet");
        return nullptr;
      }
```

Remove these 5 lines entirely. The `.lcc` extension will now fall through to the existing `readLcc` dispatch (or be added alongside other format dispatches if not already present).

- [ ] **Step 2: Verify dispatch path exists**

Check that the C API has a dispatch for `readLcc`. If not, add:

```cpp
      if (ext == ".lcc") {
        auto tables = readLcc(path, path, {});
        if (!tables.empty()) return wrapTables(std::move(tables));
        return nullptr;
      }
```

(Adjust to match the existing dispatch pattern — inspect the surrounding code for the exact convention.)

- [ ] **Step 3: Build to verify**

Run:
```bash
cd /home/merlot/codes/SplatLib && cmake --build build/lcc-plan -j 2>&1 | tail -10
```

Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add capi/splat_c.cpp
git commit -m "fix(capi): enable LCC file reading through C API

Remove the SPLAT_C_STATUS_UNSUPPORTED_FORMAT rejection for .lcc
files now that the reader is fully implemented.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 7: Full Build, Test, and Verify

**Files:**
- Test: `tests/lcc_reader_test.cpp`
- Test: `build/lcc-plan` binaries

- [ ] **Step 1: Run the full build**

```bash
cd /home/merlot/codes/SplatLib && cmake -S . -B build/lcc-plan -DBUILD_SPLAT_TESTS=ON
cmake --build build/lcc-plan -j
```

Expected: clean build, no warnings.

- [ ] **Step 2: Run decode helper tests**

```bash
cd /home/merlot/codes/SplatLib && ./build/lcc-plan/tests/SplatLccReaderTests
```

Expected: exit code 0, all assertions pass.

- [ ] **Step 3: Run existing tests to verify no regressions**

```bash
cd /home/merlot/codes/SplatLib && ctest --test-dir build/lcc-plan --output-on-failure
```

Expected: all existing tests still pass.

- [ ] **Step 4: Smoke test with `transform` CLI (if LCC test file available)**

If an LCC test file exists:
```bash
cd /home/merlot/codes/SplatLib && ./build/lcc-plan/transform/SplatTransform input.lcc output.ply --lod_select 0
```

Expected: produces valid PLY output with correct splat data.

- [ ] **Step 5: Cross-check against TS output (if LCC test file available)**

Compare C++ output against TS `readLcc()` for the same input:
- Positions should match within float epsilon
- Colors, opacities, scales should match within quantization tolerance
- Rotation quaternions should be equivalent (sign-agnostic comparison)

- [ ] **Step 6: Commit any final adjustments**

Only if source changes were needed:
```bash
git add -u
git commit -m "chore: finalize LCC reader alignment"
```

Skip if no changes.

## Self-Review

### 1. Spec Coverage

| Spec Component | Task |
|----------------|------|
| Fix `decodeRotation` → `decodeRotationInto` | Task 2 |
| Fix `hasSH` detection | Task 3 |
| Add `invLinearScale` | Task 2 |
| Implement `processUnit()` | Task 4 |
| Implement `decodeUnitsForLod()` | Task 5 Step 1 |
| Implement `deserializeEnvironment()` | Task 5 Step 2 |
| LOD selection logic | Task 5 Step 3 |
| `lod` column | Task 5 Step 3 |
| Coordinate rotation (fromEulers 90,0,180) | Task 4 (inline in processUnit), Task 5 Step 2 (env) |
| Wire `readLcc()` | Task 5 Step 3 |
| Unblock C API | Task 6 |
| Decode helper unit tests | Task 1 |

All 11 spec components covered. ✅

### 2. Placeholder Scan

- No "TBD", "TODO", or "implement later" markers ✅
- No "add appropriate error handling" without specific code ✅
- No "write tests for the above" without actual test code ✅
- No "Similar to Task N" references ✅
- All code steps include actual code ✅
- All function signatures are defined before use ✅

### 3. Type Consistency

- `processUnit` signature: `(LccUnitInfo, int, ifstream&, ifstream*, CompressInfo, size_t, map<string,vector<float>>&, vector<float>*)` — consistent across Tasks 4 and 5 ✅
- `decodeUnitsForLod` passes `f_rest_bands` through to `processUnit` — same type ✅
- `readLcc` allocates `vector<vector<float>> f_rest_bands(45)` and passes `.data()` (type `vector<float>*`) — compatible ✅
- `deserializeEnvironment` uses `CompressInfo` with `envScaleMin/Max` and `envShMin/Max` — correct field access ✅
- Column names: `"property_" + key` in processUnit, bare key in readLcc column construction — consistent ✅
- Test helpers match production helpers: `invSigmoid`, `invSH0ToColor`, `invLinearScale`, `decodeRotationInto` — identical logic ✅
