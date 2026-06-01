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
  uint32_t v = 512u | (512u << 10) | (512u << 20) | (0u << 30);
  float r0[1], r1[1], r2[1], r3[1];
  decodeRotationInto(v, r0, r1, r2, r3, 0);
  assert(near(r0[0], 0.0f, 0.01f));
  assert(near(r1[0], 1.0f, 0.01f));
  assert(near(r2[0], 0.0f, 0.01f));
  assert(near(r3[0], 0.0f, 0.01f));
}

void test_decodeRotationInto_d3_3_identity() {
  uint32_t v = 512u | (512u << 10) | (512u << 20) | (3u << 30);
  float r0[1], r1[1], r2[1], r3[1];
  decodeRotationInto(v, r0, r1, r2, r3, 0);
  assert(near(r0[0], 1.0f, 0.01f));
  assert(near(r1[0], 0.0f, 0.01f));
  assert(near(r2[0], 0.0f, 0.01f));
  assert(near(r3[0], 0.0f, 0.01f));
}

void test_decodeRotationInto_d3_1_identity() {
  uint32_t v = 512u | (512u << 10) | (512u << 20) | (1u << 30);
  float r0[1], r1[1], r2[1], r3[1];
  decodeRotationInto(v, r0, r1, r2, r3, 0);
  assert(near(r0[0], 0.0f, 0.01f));
  assert(near(r1[0], 0.0f, 0.01f));
  assert(near(r2[0], 1.0f, 0.01f));
  assert(near(r3[0], 0.0f, 0.01f));
}

void test_decodeRotationInto_output_is_unit_quaternion() {
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
