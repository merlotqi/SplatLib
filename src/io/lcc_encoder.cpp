#include "lcc_encoder.h"

#include <cmath>
#include <cstring>

namespace splat {

// ── Rotation encoding (10-10-10-2) ────────────────────────────────────────
//
// Encoding mirrors the ply2lcc reference and is the exact inverse of
// decodeRotationInto() in lcc_reader.cpp.
//
// Steps:
//  1. Normalise the input quaternion {w,x,y,z}.
//  2. Find the largest absolute component.
//  3. If the largest is negative, negate the whole quaternion.
//  4. Map from wxyz order (LCC input) to LCC index:
//       w→3, x→0, y→1, z→2
//  5. Drop the largest component; encode the remaining three.
//  6. Scale each from [-1/√2, 1/√2] to [0, 1023] (10 bits).
//  7. Pack: p0 | (p1<<10) | (p2<<20) | (lcc_idx<<30).

static constexpr float kRsqrt2 = 0.7071067811865475f;
static constexpr float kSqrt2  = 1.414213562373095f;

// Mapping from wxyz largest-index to LCC drop-index.
// wxyz idx 0 (w) → LCC 3   wxyz idx 1 (x) → LCC 0
// wxyz idx 2 (y) → LCC 1   wxyz idx 3 (z) → LCC 2
static constexpr int kWxyzToLccIdx[4] = {3, 0, 1, 2};

// Encoding order for the three stored components, by LCC drop-index.
// LCC 0 (dropped x): store y, z, w  (src indices 2, 3, 0)
// LCC 1 (dropped y): store x, z, w  (src indices 1, 3, 0)
// LCC 2 (dropped z): store x, y, w  (src indices 1, 2, 0)
// LCC 3 (dropped w): store x, y, z  (src indices 1, 2, 3)
static constexpr int kEncodeOrder[4][3] = {
    {2, 3, 0},  // LCC 0
    {1, 3, 0},  // LCC 1
    {1, 2, 0},  // LCC 2
    {1, 2, 3},  // LCC 3
};

uint32_t encodeRotation(const float rot[4]) {
  float w = rot[0], x = rot[1], y = rot[2], z = rot[3];

  // Normalise
  float len = std::sqrt(w * w + x * x + y * y + z * z);
  if (len > 0.0f) {
    w /= len;  x /= len;  y /= len;  z /= len;
  }

  // Find largest absolute component in (w, x, y, z) order
  float abs_vals[4] = {std::fabs(w), std::fabs(x), std::fabs(y), std::fabs(z)};
  int max_idx_wxyz = 0;
  for (int i = 1; i < 4; ++i) {
    if (abs_vals[i] > abs_vals[max_idx_wxyz]) max_idx_wxyz = i;
  }

  // Ensure the dropped component is positive
  float quat[4] = {w, x, y, z};
  if (quat[max_idx_wxyz] < 0.0f) {
    w = -w;  x = -x;  y = -y;  z = -z;
  }

  int lcc_idx = kWxyzToLccIdx[max_idx_wxyz];

  // Select the three components to encode
  float src[4] = {w, x, y, z};
  float enc[3];
  for (int i = 0; i < 3; ++i) {
    enc[i] = src[kEncodeOrder[lcc_idx][i]];
  }

  // Quantise each to 10 bits: map [-1/√2, 1/√2] → [0, 1023]
  auto quant10 = [](float v) -> uint32_t {
    float norm = (v + kRsqrt2) / kSqrt2;
    norm = lccClamp(norm, 0.0f, 1.0f);
    return static_cast<uint32_t>(norm * 1023.0f + 0.5f);
  };

  uint32_t p0 = quant10(enc[0]);
  uint32_t p1 = quant10(enc[1]);
  uint32_t p2 = quant10(enc[2]);

  return p0 | (p1 << 10) | (p2 << 20) | (static_cast<uint32_t>(lcc_idx) << 30);
}

// ── Color encoding ────────────────────────────────────────────────────────

uint32_t encodeColor(const float f_dc[3], float opacity) {
  auto toU8 = [](float dc) -> uint8_t {
    float color = 0.5f + kLccSH_C0 * dc;
    color = lccClamp(color, 0.0f, 1.0f);
    return static_cast<uint8_t>(color * 255.0f + 0.5f);
  };

  uint8_t r = toU8(f_dc[0]);
  uint8_t g = toU8(f_dc[1]);
  uint8_t b = toU8(f_dc[2]);
  uint8_t a = static_cast<uint8_t>(lccClamp(lccSigmoid(opacity), 0.0f, 1.0f) *
                                   255.0f + 0.5f);

  return (static_cast<uint32_t>(a) << 24) |
         (static_cast<uint32_t>(b) << 16) |
         (static_cast<uint32_t>(g) << 8) |
         static_cast<uint32_t>(r);
}

// ── Scale encoding ────────────────────────────────────────────────────────

void encodeScale(const float scale_log[3], const Eigen::Vector3f& scale_min,
                 const Eigen::Vector3f& scale_max, uint16_t out[3]) {
  for (int i = 0; i < 3; ++i) {
    float linear = std::exp(scale_log[i]);
    float range = scale_max[i] - scale_min[i];
    float norm = (range > 0.0f) ? (linear - scale_min[i]) / range : 0.0f;
    norm = lccClamp(norm, 0.0f, 1.0f);
    out[i] = static_cast<uint16_t>(norm * 65535.0f + 0.5f);
  }
}

// ── SH encoding ───────────────────────────────────────────────────────────

uint32_t encodeShTriplet(float r, float g, float b, float sh_min, float sh_max) {
  float range = sh_max - sh_min;

  auto norm = [sh_min, range](float v) -> float {
    if (range <= 0.0f) return 0.5f;
    return lccClamp((v - sh_min) / range, 0.0f, 1.0f);
  };

  uint32_t r_enc = static_cast<uint32_t>(norm(r) * 2047.0f + 0.5f);
  uint32_t g_enc = static_cast<uint32_t>(norm(g) * 1023.0f + 0.5f);
  uint32_t b_enc = static_cast<uint32_t>(norm(b) * 2047.0f + 0.5f);

  return r_enc | (g_enc << 11) | (b_enc << 21);
}

void encodeShCoef(const float f_rest[45], float sh_min, float sh_max,
                  uint32_t out[16]) {
  // f_rest layout: [R0..R14, G0..G14, B0..B14]
  const float* r_coeffs = f_rest;
  const float* g_coeffs = f_rest + 15;
  const float* b_coeffs = f_rest + 30;

  for (int i = 0; i < 15; ++i) {
    out[i] = encodeShTriplet(r_coeffs[i], g_coeffs[i], b_coeffs[i], sh_min, sh_max);
  }
  out[15] = 0;
}

// ── Full-splat encoder ────────────────────────────────────────────────────

void encodeSplat(const float pos[3], const float f_dc[3], float opacity,
                 const float scale_log[3], const float rot[4],
                 const float* f_rest, const Eigen::Vector3f& scale_min,
                 const Eigen::Vector3f& scale_max, float sh_min, float sh_max,
                 bool has_sh, uint8_t* data_out, uint8_t* sh_out) {
  // Position: 12 bytes (3 × float32)
  std::memcpy(data_out, pos, 12);
  data_out += 12;

  // Color RGBA: 4 bytes (uint32)
  uint32_t color = encodeColor(f_dc, opacity);
  std::memcpy(data_out, &color, 4);
  data_out += 4;

  // Scale: 6 bytes (3 × uint16)
  uint16_t scale_enc[3];
  encodeScale(scale_log, scale_min, scale_max, scale_enc);
  std::memcpy(data_out, scale_enc, 6);
  data_out += 6;

  // Rotation: 4 bytes (uint32)
  uint32_t rot_enc = encodeRotation(rot);
  std::memcpy(data_out, &rot_enc, 4);
  data_out += 4;

  // Normal: 6 bytes (3 × uint16, always zero for 3DGS)
  uint16_t normal_enc[3] = {0, 0, 0};
  std::memcpy(data_out, normal_enc, 6);

  // SH coefficients (optional)
  if (has_sh && f_rest != nullptr) {
    uint32_t sh_enc[16];
    encodeShCoef(f_rest, sh_min, sh_max, sh_enc);
    std::memcpy(sh_out, sh_enc, 64);
  }
}

}  // namespace splat
