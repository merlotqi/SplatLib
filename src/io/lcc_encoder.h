/**
 * @file io/lcc_encoder.h
 * @brief Encoding functions for LCC binary format (inverse of lcc_reader decode)
 *
 * Provides the compression functions that convert float splat attributes
 * into the packed binary representation used by data.bin and shcoef.bin.
 * Each function here is the exact inverse of a corresponding decode function
 * in lcc_reader.cpp.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <Eigen/Dense>

namespace splat {

/// Spherical harmonic constant C0 used for DC-to-color conversion.
constexpr float kLccSH_C0 = 0.28209479177387814f;

// ── Low-level helpers ─────────────────────────────────────────────────────

/// Logistic (sigmoid) function: maps logit-space opacity to [0,1].
inline float lccSigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

/// Clamp a float to [lo, hi].
inline float lccClamp(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// ── Per-attribute encoders ────────────────────────────────────────────────

/**
 * @brief Encode rotation quaternion into 32-bit packed format.
 *
 * 10-10-10-2 packing: the three smallest-magnitude components are
 * quantised to 10 bits each; the 2-bit index indicates which component
 * was dropped (and can be recovered via the unit-quaternion constraint).
 *
 * Input: rot[4] = {w, x, y, z}  (DataTable convention, matching
 *        decodeRotationInto output).
 * Output: uint32_t with bits [29:20]=p2, [19:10]=p1, [9:0]=p0, [31:30]=idx.
 */
uint32_t encodeRotation(const float rot[4]);

/**
 * @brief Encode SH DC coefficients + opacity into 32-bit RGBA.
 *
 * f_dc[3]  — colour in SH DC space (same convention as reader output).
 * opacity  — logit-space opacity.
 *
 * Returns packed uint32: R | (G<<8) | (B<<16) | (A<<24).
 */
uint32_t encodeColor(const float f_dc[3], float opacity);

/**
 * @brief Encode three log-space scales into 3 × uint16.
 *
 * scale_log[3] — log-space scale values (same as reader output).
 * scale_min[3], scale_max[3] — linear-space min / max bounds for dequant.
 *
 * Fills out[0..2] with the quantised uint16 values.
 */
void encodeScale(const float scale_log[3], const Eigen::Vector3f& scale_min,
                 const Eigen::Vector3f& scale_max, uint16_t out[3]);

/**
 * @brief Pack an RGB triplet into 11-10-11 bit unsigned integer.
 *
 * Range: sh_min / sh_max are used to normalise each channel independently
 * before quantising.
 */
uint32_t encodeShTriplet(float r, float g, float b, float sh_min, float sh_max);

/**
 * @brief Encode all 15 SH rest-coefficient bands into 16 × uint32 (64 bytes).
 *
 * f_rest[45]  — layout: [R0..R14, G0..G14, B0..B14] (same as reader output).
 * sh_min, sh_max — scalar range shared across all channels.
 *
 * Fills out[0..15]; out[15] is always 0 (padding).
 */
void encodeShCoef(const float f_rest[45], float sh_min, float sh_max,
                  uint32_t out[16]);

// ── Full-splat encoder ────────────────────────────────────────────────────

/**
 * @brief Encode one complete splat into the 32-byte data.bin layout and
 *        optionally the 64-byte shcoef.bin layout.
 *
 * @param pos             float[3] position in world space.
 * @param f_dc            float[3] SH DC colour coefficients.
 * @param opacity         logit-space opacity.
 * @param scale_log       float[3] log-space scales.
 * @param rot             float[4] rotation quaternion {w, x, y, z}.
 * @param f_rest          float[45] SH rest coefficients (may be null if
 *                        !has_sh).
 * @param scale_min       linear-space scale lower bounds.
 * @param scale_max       linear-space scale upper bounds.
 * @param sh_min          SH coefficient scalar lower bound.
 * @param sh_max          SH coefficient scalar upper bound.
 * @param has_sh          if true, also write 64 bytes of SH data.
 * @param data_out        destination buffer for 32-byte splat record
 *                        (must have 32 bytes available).
 * @param sh_out          destination buffer for 64-byte SH record
 *                        (must have 64 bytes available if has_sh).
 */
void encodeSplat(const float pos[3], const float f_dc[3], float opacity,
                 const float scale_log[3], const float rot[4],
                 const float* f_rest, const Eigen::Vector3f& scale_min,
                 const Eigen::Vector3f& scale_max, float sh_min, float sh_max,
                 bool has_sh, uint8_t* data_out, uint8_t* sh_out);

}  // namespace splat
