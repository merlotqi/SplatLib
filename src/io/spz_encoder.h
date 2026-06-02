/**
 * @file io/spz_encoder.h
 * @brief SPZ encoding functions — exact inverses of spz_reader decode.
 */
#pragma once

#include <cstdint>

namespace splat {

constexpr float SPZ_COLOR_SCALE = 0.15f;

inline float spzSigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/// Encode position as 3 x int24 little-endian (9 bytes per point).
void encodePosition(const float pos[3], int fractionalBits, uint8_t out[9]);

/// Encode log-scale as uint8 (3 bytes per point).
/// Formula: clamp((log_scale + 10.0) * 16.0 + 0.5, 0, 255)
void encodeScale(const float scale_log[3], uint8_t out[3]);

/// Encode SH DC color as uint8 (3 bytes per point).
/// Formula: clamp(f_dc * colorScale * 255 + 127.5 + 0.5, 0, 255)
void encodeColor(const float f_dc[3], uint8_t out[3]);

/// Encode logit-space opacity as uint8 (1 byte per point).
/// Formula: clamp(sigmoid(opacity) * 255.0 + 0.5, 0, 255)
uint8_t encodeAlpha(float opacity);

/// Encode quaternion as smallest-three packed uint32 (4 bytes per point).
/// Steps: normalize, find largest abs component, negate if needed,
/// quantize 3 smallest to 9-bit mag + 1-bit sign, pack into uint32 LE.
void encodeRotation(const float rot[4], uint8_t out[4]);

/// Encode SH coefficient as quantized uint8.
/// Formula: clamp(round((v*128+128 + bucketSize/2)/bucketSize)*bucketSize, 0, 255)
uint8_t encodeSH(float v, int bucketSize);

}  // namespace splat
