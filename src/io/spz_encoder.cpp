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

void encodeScale(const float scale_log[3], uint8_t out[3]) {
  for (int i = 0; i < 3; ++i) {
    float v = (scale_log[i] + 10.0f) * 16.0f + 0.5f;
    out[i] = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
  }
}

void encodeColor(const float f_dc[3], uint8_t out[3]) {
  for (int i = 0; i < 3; ++i) {
    float v = f_dc[i] * SPZ_COLOR_SCALE * 255.0f + 127.5f + 0.5f;
    out[i] = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
  }
}

uint8_t encodeAlpha(float opacity) {
  float v = spzSigmoid(opacity) * 255.0f + 0.5f;
  return static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
}

void encodeRotation(const float rot[4], uint8_t out[4]) {
  float w = rot[0], x = rot[1], y = rot[2], z = rot[3];
  float len = std::sqrt(w * w + x * x + y * y + z * z);
  if (len > 0.0f) { w /= len; x /= len; y /= len; z /= len; }

  float absVals[4] = {std::fabs(w), std::fabs(x), std::fabs(y), std::fabs(z)};
  unsigned largestIdx = 0;
  for (unsigned i = 1; i < 4; ++i) {
    if (absVals[i] > absVals[largestIdx]) largestIdx = i;
  }

  float quat[4] = {w, x, y, z};
  if (quat[largestIdx] < 0.0f) {
    for (int i = 0; i < 4; ++i) quat[i] = -quat[i];
  }

  constexpr float kRsqrt2 = 0.7071067811865475f;
  constexpr uint32_t kMagMask = 511u;
  uint32_t packed = largestIdx;
  for (int i = 3; i >= 0; --i) {
    if (static_cast<unsigned>(i) != largestIdx) {
      uint32_t negBit = (quat[i] < 0.0f) ? 1u : 0u;
      uint32_t mag = static_cast<uint32_t>(std::round(std::fabs(quat[i]) / kRsqrt2 * 511.0f));
      if (mag > kMagMask) mag = kMagMask;
      packed = (packed << 10) | (negBit << 9) | mag;
    }
  }

  out[0] = packed & 0xFF;
  out[1] = (packed >> 8) & 0xFF;
  out[2] = (packed >> 16) & 0xFF;
  out[3] = (packed >> 24) & 0xFF;
}

uint8_t encodeSH(float v, int bucketSize) {
  int q = static_cast<int>(std::round(v * 128.0f + 128.0f));
  q = (q + bucketSize / 2) / bucketSize * bucketSize;
  return static_cast<uint8_t>(std::clamp(q, 0, 255));
}

}  // namespace splat
