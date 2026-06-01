# LCC Reader: Align C++ Implementation with TypeScript Reference

**Date:** 2026-06-01

## Goal

Complete the C++ LCC format reader (`src/io/lcc_reader.cpp`) to match the TypeScript reference implementation (`reference/splat-transform/src/lib/readers/read-lcc.ts`). The C++ code currently has accurate metadata and index parsing but returns an empty result — the entire splat-decoding pipeline is missing. Fix all known discrepancies in decode helpers, implement the missing `processUnit` / `decodeUnitsForLod` / `deserializeEnvironment` logic, and ensure the output DataTable matches TS semantics exactly.

## Background

The LCC (Layered Compressed Cloud) format is a multi-LOD Gaussian splat container from XGRIDS. It consists of:

- `meta.lcc` — JSON metadata (attribute compression bounds, LOD levels, splat counts)
- `index.bin` — quadtree unit index with per-LOD byte offsets and sizes
- `data.bin` — packed splat records (32 bytes per splat)
- `shcoef.bin` — spherical harmonic coefficients (64 bytes per splat), present when `fileType == "Quality"`
- `environment.bin` — optional skybox splats

The C++ code at `src/io/lcc_reader.cpp` was partially ported from the TS implementation. The metadata parsing (`parseMeta`), index parsing (`parseIndexBin`), and helper functions (`decodePacked_11_10_11`, `decodeRotation`, `invSigmoid`, `invSH0ToColor`, `mixVec3`) are present and structurally correct. However:

- `readLcc()` opens the required files, parses the index, then returns an empty vector — the actual splat decoding is not implemented.
- The C `splat_c.cpp` layer returns `SPLAT_C_STATUS_UNSUPPORTED_FORMAT` for `.lcc` files.
- `decodeRotation()` has a component-mapping discrepancy vs. the TS `decodeRotationInto()`.
- `hasSH` detection logic differs from TS for the `fileType == "Portable"` case.
- The TS post-decode coordinate transform (`fromEulers(90, 0, 180)`) has no C++ equivalent.

The upstream analysis is recorded in `docs/superpowers/reference/upstream-hashes.md`.

## Design Decision

Complete the C++ LCC reader to produce semantically identical output to the TS reference. Keep the existing skeleton structure (data model in `lcc.h`, reader interface in `lcc_reader.h`, implementation in `lcc_reader.cpp`). No new files. No new dependencies.

The implementation follows the TS data flow:

```text
meta.lcc → parseMeta() → CompressInfo
index.bin → parseIndexBin() → vector<LccUnitInfo>
data.bin + shcoef.bin → processUnit() × N → pre-allocated property arrays
property arrays → DataTable columns + lod column
environment.bin → deserializeEnvironment() → optional second DataTable
```

### Scope

**In scope:**
1. Fix `decodeRotation()` to match TS `decodeRotationInto()` output mapping
2. Fix `hasSH` detection to match TS logic (`"Portable"` → false)
3. Add `invLinearScale()` helper
4. Implement `processUnit()` — decode one quadtree unit's splat data from `data.bin` and `shcoef.bin`
5. Implement `decodeUnitsForLod()` — iterate all units for a given LOD, managing write offsets
6. Implement `deserializeEnvironment()` — decode optional `environment.bin`
7. Implement LOD selection logic (`lodSelect` options)
8. Add `lod` column to output
9. Apply LCC→PlayCanvas coordinate rotation (90° around X, 180° around Z) inline during decode
10. Unblock the C API (remove `SPLAT_C_STATUS_UNSUPPORTED_FORMAT`)

**Out of scope:**
- Concurrent I/O (TS uses 16 workers; C++ uses sequential reads — acceptable for a first pass)
- LCC writer (no writer exists in TS either)
- Progress bar / logging callbacks (can be added later)

## Architecture

Changes are localized to three files:

```text
src/io/lcc_reader.cpp          ← Main implementation: fix helpers, add processUnit,
                                  decodeUnitsForLod, deserializeEnvironment, LOD logic
include/splat/models/lcc.h     ← Minor: audit ProcessUnitContext for completeness
include/splat/io/lcc_reader.h  ← No changes (API already correct)
capi/splat_c.cpp               ← Remove .lcc rejection
```

No changes to the public API. No new dependencies.

## Components

### 1. Fix `decodeRotation()` → match TS `decodeRotationInto()`

**File:** `src/io/lcc_reader.cpp`

The TS `decodeRotationInto()` writes directly to `rot0..rot3` output arrays with a component mapping that depends on `d3` (the index of the dropped component: 0=w, 1=x, 2=y, 3=z). The C++ function returns an `Eigen::Quaternionf` but maps components differently.

Replace the current `decodeRotation()` with a function that matches TS output exactly:

```cpp
// Write decoded rotation directly to output arrays at idx.
// d3 (0=w, 1=x, 2=y, 3=z) indicates which quaternion component was
// dropped during encoding and reconstructed as qw here.
// Output convention: rot_0 = w, rot_1 = x, rot_2 = y, rot_3 = z.
static void decodeRotationInto(uint32_t v,
                               float* rot0, float* rot1, float* rot2, float* rot3,
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

  // Match TS output mapping exactly
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

The old `decodeRotation()` returning `Eigen::Quaternionf` is removed (it was never called in a working code path).

### 2. Fix `hasSH` detection

**File:** `src/io/lcc_reader.cpp`

Current C++ logic:
```cpp
if (lccJson.contains("fileType")) {
  hasSH = (lccJson["fileType"] == "Quality");
} else {
  // check for "shcoef" attribute
}
```

TS logic:
```typescript
if (lccJson.fileType === 'Portable') return false;
if (lccJson.fileType === 'Quality') return true;
// fallback: check for "shcoef" attribute
```

Change C++ to match TS:
```cpp
bool hasSH = false;
if (lccJson.contains("fileType")) {
  if (lccJson["fileType"] == "Portable") {
    hasSH = false;
  } else if (lccJson["fileType"] == "Quality") {
    hasSH = true;
  } else {
    // fallback: check for "shcoef" attribute
    for (auto& attr : lccJson["attributes"]) {
      if (attr["name"] == "shcoef") { hasSH = true; break; }
    }
  }
} else {
  // No fileType field — assume SH is present (matches TS FIXME comment)
  hasSH = true;
}
```

### 3. Add `invLinearScale()` helper

**File:** `src/io/lcc_reader.cpp`

```cpp
static float invLinearScale(float v) { return std::log(v); }
```

TS: `const invLinearScale = (v: number): number => Math.log(v);`

Used to convert dequantized scale values from linear space back to log space for DataTable storage.

### 4. Implement `processUnit()`

**File:** `src/io/lcc_reader.cpp`

Decodes one quadtree unit's splat records from `data.bin` (and optionally `shcoef.bin`) into pre-allocated property arrays at the given `propertyOffset`.

**Binary record layout (data.bin, 32 bytes per splat):**

| Bytes | Type | Content |
|-------|------|---------|
| 0–3 | f32 | x position |
| 4–7 | f32 | y position |
| 8–11 | f32 | z position |
| 12 | u8 | red (0–255) |
| 13 | u8 | green (0–255) |
| 14 | u8 | blue (0–255) |
| 15 | u8 | opacity (0–255) |
| 16–17 | u16 | scale_0 (quantized, 0–65535) |
| 18–19 | u16 | scale_1 (quantized, 0–65535) |
| 20–21 | u16 | scale_2 (quantized, 0–65535) |
| 22–25 | u32 | rotation (packed: 3×10-bit + 2-bit index) |
| 26–27 | u16 | normal x |
| 28–29 | u16 | normal y |
| 30–31 | u16 | normal z |

**Binary record layout (shcoef.bin, 64 bytes per splat):**
15 consecutive u32 values, each encoding 3 SH coefficients in 11/10/11-bit packed format. SH data is read at `offset * 2` and `size * 2` relative to data.bin (because each splat has 64 bytes of SH vs. 32 bytes of base data).

**Decode pipeline per splat:**
1. Position: copy f32 values directly
2. Color: `invSH0ToColor(u8 / 255.0)` for each channel
3. Opacity: `invSigmoid(u8 / 255.0)`
4. Scale: `invLinearScale(mix(scaleMin, scaleMax, u16 / 65535.0))` for each axis
5. Rotation: `decodeRotationInto(u32, rot0, rot1, rot2, rot3, idx)`
6. Normals: copy u16 values directly (stored as raw uint16, not normalized)
7. SH (if present): 15 × u32, each decoded via `decodePacked_11_10_11` and `mix(shMin, shMax, ...)` into 3 bands (45 total f_rest values)

**Signature:**

```cpp
static void processUnit(const LccUnitInfo& info,
                        int targetLod,
                        std::ifstream& dataFile,
                        std::ifstream* shFile,
                        const CompressInfo& compressInfo,
                        size_t propertyOffset,
                        std::map<std::string, std::vector<float>>& properties,
                        std::vector<float>* properties_f_rest);
```

Implementation reads `dataBytes` (unitSplats × 32) and optionally `shBytes` (unitSplats × 64), then loops over splats decoding into `properties[property_<name>][propertyOffset + i]`.

### 5. Implement `decodeUnitsForLod()`

**File:** `src/io/lcc_reader.cpp`

Iterates all `unitInfos`, pre-computes write offsets (cumulative sum of `lods[targetLod].points`), and calls `processUnit()` for each unit. Sequential in the first pass (no threading).

```cpp
static void decodeUnitsForLod(const std::vector<LccUnitInfo>& unitInfos,
                              int targetLod,
                              std::ifstream& dataFile,
                              std::ifstream* shFile,
                              const CompressInfo& compressInfo,
                              size_t lodOffset,
                              std::map<std::string, std::vector<float>>& properties,
                              std::vector<float>* properties_f_rest);
```

### 6. Implement LOD selection logic

**File:** `src/io/lcc_reader.cpp`

Inside `readLcc()`, process the `options` vector (lodSelect):

```
if (options.empty())  → select all LODs (0..totalLevel-1)
if (options[i] < 0)   → map negative indices from end (totalLevel + lod)
filter out out-of-range LODs
```

For each selected LOD, call `decodeUnitsForLod()`, fill the `lod` column with the output LOD index, and advance `lodOffset`.

### 7. Add `lod` column

**File:** `src/io/lcc_reader.cpp`

Create a `std::vector<float>` column named `"lod"` sized to `grandTotal` splats. After decoding each LOD level, fill the corresponding range with the output LOD index (0, 1, 2, ...).

### 8. Apply LCC → PlayCanvas coordinate rotation

**File:** `src/io/lcc_reader.cpp`

The TS implementation applies `new Transform().fromEulers(90, 0, 180)` — a 90° rotation around X followed by 180° around Z. Since C++ SplatLib has no `Transform` concept on `DataTable`, apply this rotation **inline during decode** to each splat's position and rotation quaternion.

Euler (90°, 0°, 180°) in XYZ intrinsic convention yields a rotation matrix:

```
R = Rz(180°) · Rx(90°)
  = [-1  0  0]   [1  0   0]   [0  0  1]
    [ 0 -1  0] · [0  0  -1] = [0  0  1]
    [ 0  0  1]   [0  1   0]   [1  0  0]
```

Wait — this needs careful verification. The PlayCanvas `fromEulers(90, 0, 180)` documentation states the rotation order. For now, apply the known-good TS transform by rotating each position vector and quaternion during `processUnit()`:

```cpp
// Rotate position: (x, y, z) → apply R_lcc_to_pc
float rx = /* R * (px, py, pz) */;
float ry = /* ... */;
float rz = /* ... */;

// Rotate quaternion: q_out = q_lcc_to_pc * q_splat
// ...
```

The exact matrix and quaternion multiplication will be derived from the TS `Transform.fromEulers(90, 0, 180)` during implementation. A test with a known LCC file → TS output → C++ output comparison will validate correctness.

### 9. Implement `deserializeEnvironment()`

**File:** `src/io/lcc_reader.cpp`

Decodes optional `environment.bin`. Stride is 96 bytes with SH (`hasSH == true`) or 32 bytes without. Uses `envScaleMin/Max` and `envShMin/Max` from `CompressInfo` instead of the splat variants. Adds a `lod` column filled with `-1` (environment convention). The env data layout matches the splat layout but may skip the normal bytes (bytes 26–32 are zero-padded in env records).

```cpp
static std::unique_ptr<DataTable> deserializeEnvironment(
    const std::vector<uint8_t>& raw,
    const CompressInfo& compressInfo,
    bool hasSH);
```

### 10. Update `readLcc()` main flow

**File:** `src/io/lcc_reader.cpp`

Replace the current empty-result return with the full pipeline:

```text
1. Parse meta.lcc JSON → CompressInfo, splats[], totalLevel
2. Determine hasSH (fixed logic)
3. Read index.bin → vector<LccUnitInfo>
4. Open data.bin (and shcoef.bin if hasSH)
5. Resolve LOD selection from options
6. Pre-allocate property arrays (grandTotal = sum of splats[selected LODs])
7. For each selected LOD:
   a. decodeUnitsForLod(...)
   b. Fill lod column for this LOD's range
8. Build DataTable columns from properties + lod
9. Try to read environment.bin → optional second DataTable
10. Return vector<unique_ptr<DataTable>>
```

### 11. Unblock C API

**File:** `capi/splat_c.cpp`

Remove the `.lcc` rejection (lines 283-284) so the C API can dispatch to `readLcc()`.

## Breaking Changes

None. Public API unchanged. The C++ LCC reader currently returns empty results, so any caller relying on it would already be broken. This fix makes it work.

## Verification

1. **Build:** `cmake --build build --target splat`
2. **Unit test:** Create `tests/lcc_reader_test.cpp` with a minimal synthetic LCC dataset (or skip if no test LCC file is available — the reader is exercised through the transform tool)
3. **Integration:** `transform input.lcc output.ply --lod_select 0` should produce a valid PLY file with correct splat data
4. **Cross-check:** Compare C++ output against TS `readLcc()` output for the same input file (positions, colors, opacities, scales, rotations should match within float epsilon)

## References

- TS implementation: `reference/splat-transform/src/lib/readers/read-lcc.ts` (pinned at `bebac61`)
- C++ data model: `include/splat/models/lcc.h`
- C++ reader interface: `include/splat/io/lcc_reader.h`
- C++ reader (to fix): `src/io/lcc_reader.cpp`
- C API block: `capi/splat_c.cpp:283-284`
- Upstream analysis: `docs/superpowers/reference/upstream-hashes.md`
