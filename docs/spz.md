# SPZ Format Specification

SPZ is an open, compressed file format for 3D Gaussian Splatting (3DGS), developed by Niantic Labs and Adobe Inc., and standardized under the Scaniverse open-source initiative.

- **Magic Number**: `0x5053474E` ("NGSP" little-endian)
- **Latest Version**: 4
- **Repository**: [https://github.com/scaniverse/spz](https://github.com/scaniverse/spz)
- **License**: MIT

---

## 1. Version History

| Version | Header | Compression | Rotation Encoding | Status |
|---------|--------|-------------|-------------------|--------|
| 1 | 16 B | GZip | float16 positions | Never released; deprecated |
| 2 | 16 B | GZip | Quaternion first-three (8-bit × 3) | Legacy read-only |
| 3 | 16 B | GZip | Quaternion smallest-three (9-bit × 3 + 2-bit index) | Legacy read-only |
| 4 | 32 B | ZSTD multi-stream | Quaternion smallest-three | **Current** |

v4 is the canonical format. Versions 1–3 are kept for backward compatibility only and should not be produced by new writers.

---

## 2. File Structure Overview

### 2.1 v4 NGSP (Current)

```
┌─────────────────────────┐  offset 0
│  NgspFileHeader (32 B)  │
├─────────────────────────┤  offset 32
│  Extension JSON          │  (present only if flags & 0x2)
│  (variable length)       │
├─────────────────────────┤  tocByteOffset
│  TOC: numStreams ×       │
│    {compressedSize: u64, │
│     uncompressedSize:u64}│
│  = numStreams × 16 B     │
├─────────────────────────┤  tocByteOffset + tocSize
│  ZSTD stream 0           │  positions (numPoints × 9 B uncompressed)
├─────────────────────────┤
│  ZSTD stream 1           │  alphas    (numPoints × 1 B)
├─────────────────────────┤
│  ZSTD stream 2           │  colors    (numPoints × 3 B)
├─────────────────────────┤
│  ZSTD stream 3           │  scales    (numPoints × 3 B)
├─────────────────────────┤
│  ZSTD stream 4           │  rotations (numPoints × 4 B)
├─────────────────────────┤
│  ZSTD stream 5           │  sh        (numPoints × shDim × 3 B)
│  (only if shDegree > 0)  │
└─────────────────────────┘
```

Streams with zero uncompressed size are skipped entirely (they do not appear in the TOC or as data chunks). The stream count in `numStreams` reflects only the non-empty streams.

### 2.2 v2–v3 Legacy (GZip)

```
┌──────────────────────┐  offset 0 (after GZip decompression)
│ LegacyHeader (16 B)  │
├──────────────────────┤  offset 16
│ positions            │  numPoints × 9 B  (3 × int24 LE per point)
├──────────────────────┤
│ alphas               │  numPoints × 1 B
├──────────────────────┤
│ colors               │  numPoints × 3 B
├──────────────────────┤
│ scales               │  numPoints × 3 B
├──────────────────────┤
│ rotations            │  numPoints × 3 B (v2) or × 4 B (v3)
├──────────────────────┤
│ sh                   │  numPoints × shDim × 3 B
└──────────────────────┘
```

Entire payload is compressed with GZip (RFC 1952). The GZip magic bytes (`0x1F 0x8B`) are the primary detection mechanism for the legacy path.

---

## 3. Headers

### 3.1 NgspFileHeader (v4, 32 bytes)

```
Offset  Size  Type     Field           Description
──────  ────  ────     ─────           ───────────
0       4     u32      magic           0x5053474E ("NGSP")
4       4     u32      version         4
8       4     u32      numPoints       Total number of Gaussians
12      1     u8       shDegree        SH degree (0–4)
13      1     u8       fractionalBits  Position fixed-point precision (default 12)
14      1     u8       flags           bit0: antialiased, bit1: hasExtensions
15      1     u8       numStreams      Number of ZSTD-compressed attribute streams
16      4     u32      tocByteOffset   Byte offset from file start to TOC
20      12    u8[12]   reserved        Zero-filled
```

### 3.2 Legacy Header (v1–v3, 16 bytes)

```
Offset  Size  Type     Field           Description
──────  ────  ────     ─────           ───────────
0       4     u32      magic           0x5053474E
4       4     u32      version         1, 2, or 3
8       4     u32      numPoints       Total number of Gaussians
12      1     u8       shDegree        SH degree (0–4)
13      1     u8       fractionalBits  Position precision
14      1     u8       flags           bit0: antialiased
15      1     u8       reserved        Zero
```

---

## 4. Attribute Encoding

All attribute data is stored as **non-interleaved** `uint8_t` arrays. Each attribute occupies a contiguous sub-buffer within its stream.

### 4.1 Position

**Encoding**: 24-bit signed fixed-point, little-endian byte order.

Per coordinate (x, y, z individually):

```
Encode:  int32_t fixed = round(value × 2^fractionalBits)
         buf[0] = fixed & 0xFF
         buf[1] = (fixed >> 8) & 0xFF
         buf[2] = (fixed >> 16) & 0xFF

Decode:  int32_t raw = buf[0] | (buf[1] << 8) | (buf[2] << 16)
         if (raw & 0x800000) raw |= 0xFF000000   // sign-extend 24→32 bits
         value = (float)raw / 2^fractionalBits
```

**Bytes per point**: 9 (3 coordinates × 3 bytes each)

**Default precision**: `fractionalBits = 12` → ~0.25 mm resolution at unit scale.

### 4.2 Scale

**Encoding**: Logarithmic scale quantized to uint8.

```
Encode:  byte = clamp(round((log_scale + 10.0) × 16.0), 0, 255)

Decode:  log_scale = byte / 16.0 - 10.0
```

**Bytes per point**: 3 (scale_x, scale_y, scale_z)

The log scale is the natural logarithm of the Gaussian's standard deviation along each axis: `σ_i = exp(scale_i)`.

### 4.3 Color (SH DC)

**Encoding**: Spherical Harmonics degree-0 coefficients as wide-range RGB.

```
Encode:  byte = clamp(round(f_dc × colorScale × 255 + 127.5), 0, 255)

Decode:  f_dc = (byte / 255.0 - 0.5) / colorScale
```

**Bytes per point**: 3 (R, G, B)

`colorScale = 0.15` allows the encoded range to exceed [0, 1] slightly (up to ~1.7), accommodating SH DC coefficients that represent colors outside the displayable gamut — higher-order SH bands can bring them back into range during rendering.

To convert between f_dc and linear RGB:

```
RGB = 0.5 + C0 × f_dc        where C0 = 0.28209479177387814 (SH constant)
f_dc = (RGB - 0.5) / C0
```

### 4.4 Alpha (Opacity)

**Encoding**: Sigmoid-activated opacity.

```
Encode:  byte = clamp(round(sigmoid(opacity) × 255), 0, 255)
         where sigmoid(x) = 1 / (1 + exp(-x))

Decode:  norm = byte / 255.0
         opacity = log(norm / (1 - norm))     // inverse sigmoid
```

**Bytes per point**: 1

The stored opacity is in **logit space** (inverse logistic). `opacity = 0` → alpha ≈ 0.5 after sigmoid; `opacity → +∞` → alpha → 1.

### 4.5 Rotation (Quaternion)

**v3+ Encoding**: Smallest-three quaternion packing (4 bytes per point).

The rotation quaternion `{w, x, y, z}` is normalized, then the component with the largest absolute value is identified as the "dropped" component. The remaining three components are quantized to 9-bit magnitude + 1-bit sign. Since a unit quaternion satisfies `w²+x²+y²+z²=1`, the dropped component can be recovered from the other three.

**Encode**:
```
1. Normalize quaternion:  len = sqrt(w²+x²+y²+z²);  {w,x,y,z} /= len
2. Find largest absolute component index:  iMax ∈ {0(w),1(x),2(y),3(z)}
3. If quat[iMax] < 0, negate entire quaternion (ensures dropped component > 0)
4. For each of the 3 non-dropped components (iterating from 3 down to 0):
     mag = clamp(round(|q[i]| / (1/√2) × 511), 0, 511)
     sign = (q[i] < 0) ? 1 : 0
     pack = (pack << 10) | (sign << 9) | mag
5. Final uint32:  upper 2 bits = iMax, lower 30 bits = 3×(sign+mag)
```

**Decode**:
```
1. iMax = packed >> 30
2. For i = 3 down to 0, i ≠ iMax:
     mag = packed & 511
     sign = (packed >> 9) & 1
     q[i] = (1/√2) × mag / 511 × (sign ? -1 : 1)
     sumSq += q[i]²
     packed >>= 10
3. q[iMax] = sqrt(1 - sumSq)
```

**v2 Encoding**: First-three quaternion packing (3 bytes, legacy).
```
Encode:  byte[i] = clamp(round((q[i] / 127.5 + 1) × 127.5), 0, 255)   for i=x,y,z
Decode:  q[i] = byte[i] / 127.5 - 1.0          for i=x,y,z
         q[w] = sqrt(1 - qx² - qy² - qz²)
```

### 4.6 Spherical Harmonics

**Encoding**: Signed float → uint8 with configurable quantization.

```
Encode:  q = round(value × 128 + 128)                    // map [-1,+1] → [0,255]
         q = (q + bucketSize/2) / bucketSize × bucketSize // quantize
         byte = clamp(q, 0, 255)

Decode:  value = (byte - 128.0) / 128.0
```

**Bytes per point**: `shDim × 3`, where `shDim` depends on `shDegree`:

| shDegree | shDim | Bytes/point | Coeffs (R,G,B each) |
|----------|-------|-------------|---------------------|
| 0 | 0  | 0  | — |
| 1 | 3  | 9  | SH band 1: 3 coeffs × 3 channels |
| 2 | 8  | 24 | SH bands 1+2: 8 coeffs × 3 channels |
| 3 | 15 | 45 | SH bands 1–3: 15 coeffs × 3 channels |
| 4 | 24 | 72 | SH bands 1–4: 24 coeffs × 3 channels |

**Data layout**: The inner (fastest-varying) axis is the color channel (R, G, B), and the outer axis is the SH coefficient index. For shDegree=3 the order is:

```
f_rest_0 = sh1n1_r,  f_rest_1 = sh1n1_g,  f_rest_2 = sh1n1_b,
f_rest_3 = sh10_r,   f_rest_4 = sh10_g,   f_rest_5 = sh10_b,
f_rest_6 = sh1p1_r,  f_rest_7 = sh1p1_g,  f_rest_8 = sh1p1_b,
... (bands 2 and 3 follow)
```

**Quantization**:
- `sh1Bits` (default 5): quantization bits for degree-1 coefficients (first 9 slots)
- `shRestBits` (default 4): quantization bits for degree-2+ coefficients
- `bucketSize = 1 << (8 - bits)`: e.g., sh1Bits=5 → bucketSize=8 → 32 quantization levels

Quantization is applied at **encode time** only to improve ZSTD compression by reducing entropy. The decoder does not need these values — the zero-filled lower bits naturally emerge after decompression.

---

## 5. Per-Point Attribute Summary

| Attribute | Uncompressed Bytes | Type | Range | Formula |
|-----------|-------------------|------|-------|---------|
| Position | 9 (3×3) | int24 LE | arbitrary | `round(v × 2^12)` |
| Alpha | 1 | uint8 | [0,255] | `sigmoid(opacity) × 255` |
| Color | 3 | uint8 | [0,255] | `(f_dc × 0.15 + 0.5) × 255` |
| Scale | 3 | uint8 | [0,255] | `(log_scale + 10) × 16` |
| Rotation | 4 | uint32 LE | 10-10-10-2 | smallest-three packing |
| SH | shDim×3 | uint8 | [0,255] | `(coeff × 128 + 128)` |

**Total per point (no SH)**: `9 + 1 + 3 + 3 + 4 = 20 bytes` uncompressed.

**Total per point (shDegree=3)**: `20 + 45 = 65 bytes` uncompressed.

---

## 6. Coordinate System

SPZ stores data in the **RUB (Right, Up, Back)** coordinate system, matching the Three.js convention:

```
+X = Right
+Y = Up
+Z = Back (toward viewer in a right-handed system)
```

Important reference points:

| System | Code | Convention |
|--------|------|------------|
| RUB | 4 | SPZ storage format; Three.js |
| RDF | 6 | PLY format standard |
| LUF | 7 | GLB / glTF format |
| RUF | 8 | Unity |

The 4-bit encoding scheme: `(rightFlag, upFlag, forwardFlag, swapYZ)`.

**Within-family** conversions (same `swapYZ` bit) use sign flips only.
**Cross-family** conversions (different `swapYZ` bit) additionally apply `R_x(±π/2)` analytic rotation to positions, quaternions, and SH coefficients.

For details, see `splat/maths/coordinate-converter.h` and the reference implementation in `splat-types.h`.

---

## 7. File Detection

```
1. Read first 4 bytes as uint32_t LE
2. If value == 0x5053474E:
     Read next 4 bytes as uint32_t LE (version)
     If version >= 4: parse as NGSP v4 with ZSTD streams
     Else: parse as legacy format (v2–v3)
3. Else if first 2 bytes == 0x1F 0x8B (GZip magic):
     Decompress with GZip, then parse as legacy format
4. Else: not a valid SPZ file
```

---

## 8. References

- [SPZ GitHub Repository](https://github.com/nianticlabs/spz)
- [SPZ: Open-Source Gaussian Splat Format](https://scaniverse.com/news/spz-open-source-gaussian-splat-file-format)
- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- SplatLib implementation: `src/io/spz_reader.cpp`, `src/io/spz_encoder.cpp`
