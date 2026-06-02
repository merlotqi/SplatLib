# SPZ Format Alignment Design

## Context

SplatLib has a partial SPZ reader (`src/io/spz_reader.cpp`) that supports only the legacy v2-v3 GZip single-stream format. The reference SPZ library (Niantic/Adobe, `reference/spz/`) has evolved to v4 with ZSTD multi-stream compression, a complete writer, and 16-coordinate-system conversion. This design describes aligning SplatLib's SPZ implementation with the reference.

### Source Files

| Role | Path |
|------|------|
| Reference implementation | `reference/spz/src/cc/load-spz.cc` (1500 lines) |
| Reference types | `reference/spz/src/cc/splat-types.h` |
| SplatLib current reader | `src/io/spz_reader.cpp` (209 lines) |
| SplatLib current API | `include/splat/io/spz_reader.h` |

### Existing Gap Summary

| Capability | SplatLib | Reference |
|-----------|----------|-----------|
| v1-v3 read (GZip) | v2-v3 only | v1-v3 |
| v4 read (ZSTD NGSP) | ❌ | ✅ |
| SPZ write | ❌ | ✅ |
| Coordinate conversion | ❌ | 16 systems |
| PLY bridge | separate `ply_reader/writer` | integrated in SPZ lib |

**Core decode correctness**: SplatLib's v2-v3 attribute decoding (position int24, scale, color, alpha, rotation, SH) is mathematically identical to the reference. The gaps are purely in format version coverage and missing features.

---

## Architecture

```
                        ┌─────────────────────────┐
                        │   CoordinateConverter    │  Iter 3
                        │   maths/coordinate-      │
                        │   converter.cpp          │
                        └─────────┬───────────────┘
                                  │ consumed by SPZ r/w
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
  ┌───────▼──────┐     ┌─────────▼─────────┐    ┌────────▼────────┐
  │ spz_reader   │     │  spz_encoder (内部) │    │  spz_writer     │
  │ v2-v3 (GZip) │     │  encodePosition    │    │  DataTable →    │
  │ v4   (ZSTD)  │     │  encodeColor       │    │  PackedGaussians│
  └──────────────┘     │  encodeScale       │    │  → ZSTD streams │
      Iter 1           │  encodeRotation    │    │  → .spz file    │
                       │  encodeSH          │    └────────────┬───┘
                       └────────────────────┘               │
                              Iter 2              ┌────────▼────────┐
                                                  │  spz_writer.h   │
                                                  │  (public API)    │
                                                  └─────────────────┘
```

### Iteration Plan

| Iter | Scope | New Files | Modified Files | Est. LOC |
|------|-------|-----------|---------------|----------|
| 1 | v4 NGSP Reader | — | `spz_reader.cpp` | +200 |
| 2 | SPZ Writer | `spz_encoder.h/cpp`, `spz_writer.h/cpp` | — | +500 |
| 3 | CoordinateConverter | `maths/coordinate-converter.h/cpp` | — | +400 |

Each iteration is independently testable and committable.

---

## Iteration 1: v4 NGSP Reader

### Objective

Add v4 ZSTD multi-stream format support to the existing `readSpz()`. The v2-v3 GZip path is preserved unchanged.

### v4 File Layout

```
Offset       Size    Field
0            4       magic = 0x5053474E ("NGSP")
4            4       version (≥4)
8            4       numPoints
12           1       shDegree
13           1       fractionalBits
14           1       flags (bit0=antialiased, bit1=hasExtensions)
15           1       numStreams
16           4       tocByteOffset
20           12      reserved
── 32 bytes total ──
32           ?       Extension JSON (present if flags & 0x2)
tocByteOffset       TOC: numStreams × {compressedSize:u64, uncompressedSize:u64}
TOC+?               ZSTD stream 0 (positions, numPoints×9 bytes uncompressed)
                    ZSTD stream 1 (alphas,    numPoints×1 bytes)
                    ZSTD stream 2 (colors,    numPoints×3 bytes)
                    ZSTD stream 3 (scales,    numPoints×3 bytes)
                    ZSTD stream 4 (rotations, numPoints×4 bytes)
                    ZSTD stream 5 (sh,        numPoints×shDim×3 bytes)
```

Streams with zero uncompressed size are omitted. Stream count in TOC must match `numStreams`.

### Header Detection

Add to `readSpz()` before the GZip check:

```cpp
// 1. Check NGSP magic → v4 ZSTD multi-stream
uint32_t magic;
std::memcpy(&magic, buffer.data(), 4);
if (magic == 0x5053474E) {
    return readSpzV4(buffer);  // new path
}
// 2. Check GZip magic → existing v2-v3 path (unchanged)
if (buffer.size() > 2 && buffer[0] == 0x1F && buffer[1] == 0x8B) {
    buffer = decompressGZIP(buffer);
}
// 3. Existing 16-byte header parsing (unchanged)
```

### ZSTD Dependency

SplatLib already links `ZLIB::ZLIB`. Add `zstd` via vcpkg:

```cmake
# CMakeLists.txt
find_package(zstd REQUIRED)
target_link_libraries(splat PUBLIC ZSTD::ZSTD)
```

### Stream Decompression

```cpp
bool decompressNgspStreams(
    const uint8_t* data, size_t size,
    const NgspFileHeader& header,
    std::vector<std::pair<uint8_t*, size_t>> dests);
```

- Parse TOC: for each stream, read `{compressedSize, uncompressedSize}` at `tocByteOffset + i*16`
- Validate total compressed size matches `size - (tocByteOffset + tocSize)`
- For each stream: `ZSTD_decompress(dest, destSize, src, compressedSize)`
- Uncompressed data is written directly into pre-allocated DataTable columns (zero-copy)

### Attribute Decode

Reuse existing v2-v3 decode logic unchanged. The per-attribute decoding functions are identical between v3 and v4 — only the compression wrapper differs. Extract shared decode into internal helpers:

```cpp
// Shared between v2-v3 and v4 paths
static void decodePositions(const uint8_t* posBase, int fractionalBits,
                            uint32_t numSplats, std::vector<float*>& cols);
static void decodeScales(const uint8_t* scaleBase, uint32_t numSplats,
                          std::vector<float*>& cols);
// ... same for rotations, colors, alphas, sh
```

### Changes

| File | Change |
|------|--------|
| `src/io/spz_reader.cpp` | Add v4 detection, NgspFileHeader parse, TOC, ZSTD decompress, extract shared decode helpers |
| `CMakeLists.txt` | Add `find_package(zstd)` and link |
| `vcpkg.json` or manifest | Add `zstd` dependency |

---

## Iteration 2: SPZ Writer

### Objective

Implement `writeSpz()` — encode DataTable columns into v4 ZSTD multi-stream .spz files.

### New Files

| File | Purpose |
|------|---------|
| `include/splat/io/spz_writer.h` | Public API: `SpzWriteOptions`, `writeSpz()` |
| `src/io/spz_encoder.h` | Internal encoding function declarations |
| `src/io/spz_encoder.cpp` | Encoding implementation |
| `src/io/spz_writer.cpp` | Pipeline orchestration + file output |

### Public API

```cpp
// include/splat/io/spz_writer.h
namespace splat {

struct SpzWriteOptions {
  int fractionalBits = 12;      // position fixed-point precision
  uint8_t sh1Bits = 5;         // SH degree-1 quantization bits
  uint8_t shRestBits = 4;      // SH degree-2+ quantization bits
};

void writeSpz(const std::filesystem::path& filename,
              const DataTable& dataTable,
              const SpzWriteOptions& options = {});

}  // namespace splat
```

### Encoding Functions (exact inverses of reader decodes)

All in `src/io/spz_encoder.cpp`, internal (not public API):

#### encodePosition
```
Input:  float[3] position
Output: uint8_t[9]  (3 × int24 little-endian)
Formula: int32_t fixed = round(v × 2^fractionalBits)
         out[0] = fixed & 0xFF
         out[1] = (fixed >> 8) & 0xFF
         out[2] = (fixed >> 16) & 0xFF
```

#### encodeScale
```
Input:  float[3] log-scale
Output: uint8_t[3]
Formula: out[i] = clamp((log_scale[i] + 10.0f) × 16.0f + 0.5f, 0, 255)
```
Inverse of reader's `scale/16.0 - 10.0`.

#### encodeColor
```
Input:  float[3] f_dc (SH DC coefficients)
Output: uint8_t[3]
Formula: out[i] = clamp(f_dc[i] × 0.15f × 255.0f + 127.5f + 0.5f, 0, 255)
```
Inverse of reader's `(c/255.0 - 0.5) / 0.15`. `colorScale = 0.15` matches reference.

#### encodeAlpha
```
Input:  float opacity (logit space)
Output: uint8_t
Formula: out = clamp(sigmoid(opacity) × 255.0f + 0.5f, 0, 255)
         sigmoid(x) = 1/(1+exp(-x))
```
Inverse of reader's `log(α/(1-α))`.

#### encodeRotation
```
Input:  float[4] quaternion {w, x, y, z}
Output: uint8_t[4] (packed uint32 little-endian)
```
Matches reference `packQuaternionSmallestThree()`:
1. Normalize quaternion
2. Find largest absolute component index (0=w, 1=x, 2=y, 3=z)
3. If largest is negative, negate entire quaternion
4. For the 3 non-largest components: quantize to 9-bit magnitude + 1-bit sign
   - `mag = round(|q[i]| / (1/√2) × 511)`, clamped to [0,511]
   - `sign = q[i] < 0 ? 1 : 0`
   - Pack as `(sign << 9) | mag`
5. Pack into uint32: low 30 bits = 3×(sign+mag), high 2 bits = largest index
6. Store as 4 uint8 LE

Inverse of reader's `unpackQuaternionSmallestThree`.

#### encodeSH
```
Input:  float[shDim×3] SH coefficients (R,G,B interleaved)
        uint8_t sh1Bits, shRestBits
Output: uint8_t[shDim×3]
Formula: For each coefficient:
           normalized = (v + 1.0) / 2.0  (map [-1,1] → [0,1])
           bucketSize = 1 << (8 - bits)   (bits = sh1Bits for band 0-1, shRestBits for band 2+)
           quantized = round((v * 128.0 + 128.0 + bucketSize/2) / bucketSize) * bucketSize
           out = clamp(quantized, 0, 255)
```
Inverse of reader's `(val - 128.0) / 128.0`. Quantization reduces entropy for better ZSTD compression.

### Writer Pipeline

```
writeSpz(filename, dataTable, options)
  │
  ├── 1. Validate columns: x, y, z, scale_0..2, f_dc_0..2, opacity, rot_0..3 required;
  │       f_rest_0..N optional
  │
  ├── 2. Determine shDegree from f_rest column count (0/3/8/15/24 coeffs → degree 0/1/2/3/4)
  │
  ├── 3. Allocate PackedGaussians buffers:
  │       positions  = numPoints × 9
  │       alphas     = numPoints × 1
  │       colors     = numPoints × 3
  │       scales     = numPoints × 3
  │       rotations  = numPoints × 4
  │       sh         = numPoints × shDim × 3  (if shDegree > 0)
  │
  ├── 4. Per-splat encode (single-threaded loop):
  │       encodePosition → positions buffer
  │       encodeScale    → scales buffer
  │       encodeColor    → colors buffer
  │       encodeAlpha    → alphas buffer
  │       encodeRotation → rotations buffer
  │       encodeSH       → sh buffer (if applicable)
  │
  ├── 5. ZSTD compress each non-zero buffer
  │       → chunks[i], uncompressedSizes[i]
  │
  ├── 6. Build NgspFileHeader (32 bytes)
  │
  └── 7. Write file:
         [header][TOC = streams×16B][chunk0][chunk1]...
```

### Edge Cases

- **Portable (no SH)**: `shDegree=0`, `shDim=0`, `sh` buffer omitted from streams
- **Single point**: encoding still produces valid ZSTD stream
- **All-zero scale/color**: handled correctly by clamp in encode functions

---

## Iteration 3: CoordinateConverter

### Objective

Port the reference `CoordinateSystem` enum and `CoordinateConverter` logic as a reusable module in `src/maths/`. This module has zero SPZ dependencies — it operates on raw float arrays.

### New Files

| File | Purpose |
|------|---------|
| `include/splat/maths/coordinate-converter.h` | `CoordinateSystem` enum, `CoordinateConverter` struct, factory function |
| `src/maths/coordinate-converter.cpp` | `makeCoordinateConverter()`, `convertPosition()`, `convertRotation()`, `convertSH()` |

### Coordinate System Enum

16 systems encoded as 4 bits: `(right, up, forward, swapYZ)`. Matches reference `CoordinateSystem`:

```cpp
enum class CoordinateSystem : uint32_t {
  UNSPECIFIED = 0,
  LDB = 1,  // Left  Down Back
  RDB = 2,  // Right Down Back
  LUB = 3,  // Left  Up   Back
  RUB = 4,  // Right Up   Back  ← SPZ/Three.js default
  LDF = 5,  // Left  Down Front
  RDF = 6,  // Right Down Front ← PLY default
  LUF = 7,  // Left  Up   Front ← GLB
  RUF = 8,  // Right Up   Front ← Unity
  // family-rotated variants (swapYZ=1):
  LFD = 9, RFD = 10, LFU = 11, RFU = 12,
  LBD = 13, RBD = 14, LBU = 15, RBU = 16,
};
```

### CoordinateConverter Structure

```cpp
struct CoordinateConverter {
  // Within-family: simple sign flips
  std::array<float, 3> flipP;        // position flip per axis
  std::array<float, 3> flipQ;        // quaternion x,y,z flip (w never flipped)
  std::array<float, 24> flipSh;      // per-coefficient SH sign flip

  // Cross-family: rotation + flip composed into single function per attribute
  // These are non-null only for cross-family conversions.
  void (*rotFlipPos)(float*);            // position: rotate then flip
  void (*rotFlipQuat)(float*);           // quaternion: rotate then flip
  void (*rotFlipShBands[4])(float*);     // SH per band (max degree 4)
};
```

### Conversion Logic

#### Within-Family (same bit3)

Both systems share the same Right/Left ↔ Up/Down ↔ Front/Back axis ordering — only signs differ.

For each axis where `from[i] != to[i]`: flip the sign.

- **Position**: multiply `pos[i] *= -1` for mismatched axes
- **Quaternion**: multiply `quat[i] *= -1` for mismatched axes; w unchanged
- **SH**: per-coefficient flip table computed from axis parity

#### Cross-Family (different bit3)

The Right/Left ↔ Up/Down axis basis is swapped (e.g., RUB ↔ RDF). This requires R_x(±π/2) rotation before the sign flip.

- **Position**: swap y↔z with sign adjustment: `R_x(+π/2): (x, y, z) → (x, -z, y)`
- **Quaternion**: pre-multiply by `R_x(+π/2)` quaternion = `(√2/2, √2/2, 0, 0)`
- **SH**: apply band-l analytic rotation matrices (`kAnalyticRotatePlusPiHalfAboutXTable[l]`)

The reference provides pre-computed SH rotation matrices for bands l=1..4 (each operates on 2l+1 coefficients). These are pure linear transformations with `√3`, `√5`, `√7`, `√15`, `√35` constants — directly portable.

### Factory Function

```cpp
CoordinateConverter makeCoordinateConverter(
    CoordinateSystem from,
    CoordinateSystem to,
    int shDegree);
```

- If `from == to` or either is UNSPECIFIED: identity converter (all flips=1, no rotate functions)
- If same family (bit3 equal): compute sign flips only
- If cross-family: compute rotation function composition first, then sign flips

### Integration Points

In SPZ reader (Iter 1):
```cpp
// After decode, optionally convert
if (options.coordinateSystem != CoordinateSystem::UNSPECIFIED) {
    auto conv = makeCoordinateConverter(CoordinateSystem::RUB, options.coordinateSystem, shDegree);
    for each splat: conv.convertPosition, conv.convertRotation, conv.convertSH
}
```

In SPZ writer (Iter 2):
```cpp
// Before encode, optionally convert to RUB
if (options.sourceCoordinateSystem != CoordinateSystem::RUB) {
    auto conv = makeCoordinateConverter(options.sourceCoordinateSystem, CoordinateSystem::RUB, shDegree);
    for each splat: conv.convertPosition, conv.convertRotation, conv.convertSH
}
```

Default behavior: no conversion (UNSPECIFIED) — preserves backward compatibility.

---

## Verification

### Iter 1

- **Unit test**: Round-trip a reference v4 .spz file → DataTable → verify column values match known reference
- **Unit test**: Test with shDegree=0 (no SH), shDegree=3 (45 coeffs)
- **Regression**: All existing v2-v3 .spz files still load correctly

### Iter 2

- **Unit test**: Round-trip DataTable → writeSpz → readSpz → verify all float columns match within quantization tolerance
- **Unit test**: Test with and without SH
- **Unit test**: Test single-splat edge case

### Iter 3

- **Unit test**: Identity conversion (RUB→RUB) is no-op
- **Unit test**: RUB↔RDF round-trip: position and quaternion preserved
- **Unit test**: Cross-family SH rotation: verify band coefficients match reference
- **Unit test**: All 16 coordinate systems convert to RUB without crash

---

## Dependencies

| Dependency | Needed By | Source |
|-----------|-----------|--------|
| `zstd` | Iter 1, 2 | vcpkg `zstd` port |
| `zlib` | Iter 1 | Already linked |
| `Eigen` | Iter 3 | Already linked (vec3/quat math) |

---

## What This Design Does NOT Cover

- **Parallel compression**: Reference uses `std::async` for ZSTD streams. Start single-threaded; parallelize as optimization later.
- **Float16 legacy path**: v1 was never released. Not worth implementing.
- **SPZ extensions**: `SPZ_BUILD_EXTENSIONS` in reference adds custom metadata. Out of scope for initial alignment.
- **PLY bridge**: SplatLib already has separate `ply_reader/writer`. No need to duplicate reference's `loadSplatFromPly/saveSplatToPly`.
- **GZip writer**: v4 always uses ZSTD. Legacy GZip write path not needed.
