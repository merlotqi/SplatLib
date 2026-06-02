/**
 * @file lcc_writer_test.cpp
 * @brief Unit tests for LCC writer encoding functions — round-trip verification.
 */

#include "../src/io/lcc_encoder.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr float kSH_C0 = 0.28209479177387814f;
constexpr float SQRT_2 = 1.41421356237f;
constexpr float SQRT_2_INV = 0.70710678118f;

float invSigmoid(float v) { return -std::log((1.0f - v) / v); }

float invSH0ToColor(float v) { return (v - 0.5f) / kSH_C0; }

void decodePacked_11_10_11(float& r, float& g, float& b, uint32_t enc) {
  r = static_cast<float>(enc & 2047) / 2047.0f;
  g = static_cast<float>((enc >> 11) & 1023) / 1023.0f;
  b = static_cast<float>((enc >> 21) & 2047) / 2047.0f;
}

void decodeRotationInto(uint32_t v, float& r0, float& r1, float& r2, float& r3) {
  float d0 = static_cast<float>(v & 1023) / 1023.0f;
  float d1 = static_cast<float>((v >> 10) & 1023) / 1023.0f;
  float d2 = static_cast<float>((v >> 20) & 1023) / 1023.0f;
  uint32_t d3 = (v >> 30) & 3;

  float qx = d0 * SQRT_2 - SQRT_2_INV;
  float qy = d1 * SQRT_2 - SQRT_2_INV;
  float qz = d2 * SQRT_2 - SQRT_2_INV;
  float sum = std::min(1.0f, qx * qx + qy * qy + qz * qz);
  float qw = std::sqrt(1.0f - sum);

  if (d3 == 0)       { r0 = qz; r1 = qw; r2 = qx; r3 = qy; }
  else if (d3 == 1)  { r0 = qz; r1 = qx; r2 = qw; r3 = qy; }
  else if (d3 == 2)  { r0 = qz; r1 = qx; r2 = qy; r3 = qw; }
  else               { r0 = qw; r1 = qx; r2 = qy; r3 = qz; }
}

void check(bool cond, const char* msg) {
  if (!cond) { fprintf(stderr, "FAIL: %s\n", msg); std::exit(1); }
}

}  // namespace

// ── encodeRotation round-trip ─────────────────────────────────────────────

static void testEncodeRotationIdentity() {
  float rot[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  uint32_t enc = splat::encodeRotation(rot);

  float w, x, y, z;
  decodeRotationInto(enc, w, x, y, z);

  float len = std::sqrt(w*w + x*x + y*y + z*z);
  check(std::fabs(w/len - 1.0f) < 0.002f, "encodeRotation identity");
  printf("PASS: encodeRotation identity\n");
}

static void testEncodeRotationArbitrary() {
  float r0 = 0.5f, r1 = 0.5f, r2 = 0.5f, r3 = 0.5f;
  float len = std::sqrt(r0*r0 + r1*r1 + r2*r2 + r3*r3);
  float rot[4] = {r0/len, r1/len, r2/len, r3/len};

  uint32_t enc = splat::encodeRotation(rot);

  float w, x, y, z;
  decodeRotationInto(enc, w, x, y, z);

  float dot = w*rot[0] + x*rot[1] + y*rot[2] + z*rot[3];
  check(std::fabs(std::fabs(dot) - 1.0f) < 0.002f, "encodeRotation arbitrary");
  printf("PASS: encodeRotation arbitrary\n");
}

static void testEncodeRotationAxes() {
  float s2 = std::sin(M_PI / 4.0);
  float c2 = std::cos(M_PI / 4.0);

  struct { float w,x,y,z; const char* n; } cases[] = {
    {c2, s2, 0, 0, "90 X"},
    {c2, 0, s2, 0, "90 Y"},
    {c2, 0, 0, s2, "90 Z"},
    {0, 1, 0, 0, "180 X"},
  };

  for (auto& tc : cases) {
    float rot[4] = {tc.w, tc.x, tc.y, tc.z};
    uint32_t enc = splat::encodeRotation(rot);

    float w, x, y, z;
    decodeRotationInto(enc, w, x, y, z);

    float dot = w*tc.w + x*tc.x + y*tc.y + z*tc.z;
    check(std::fabs(std::fabs(dot) - 1.0f) < 0.002f, tc.n);
    printf("PASS: encodeRotation %s\n", tc.n);
  }
}

// ── encodeColor round-trip ────────────────────────────────────────────────

static void testEncodeColorGray() {
  float f_dc[3] = {0.0f, 0.0f, 0.0f};
  float opacity = 0.0f;

  uint32_t enc = splat::encodeColor(f_dc, opacity);
  uint8_t r8 = enc & 0xFF, g8 = (enc>>8)&0xFF, b8 = (enc>>16)&0xFF;

  check(r8 >= 126 && r8 <= 129 && g8 >= 126 && g8 <= 129 && b8 >= 126 && b8 <= 129,
        "encodeColor gray");
  printf("PASS: encodeColor gray\n");
}

static void testEncodeColorRoundTrip() {
  float f_dc[3] = {0.3f, -0.2f, 0.5f};
  float opacity = 1.5f;

  uint32_t enc = splat::encodeColor(f_dc, opacity);

  uint8_t r8 = enc & 0xFF, g8 = (enc>>8)&0xFF, b8 = (enc>>16)&0xFF;
  float f_dc_r = invSH0ToColor(r8 / 255.0f);
  float f_dc_g = invSH0ToColor(g8 / 255.0f);
  float f_dc_b = invSH0ToColor(b8 / 255.0f);

  check(std::fabs(f_dc_r - f_dc[0]) < 0.01f &&
        std::fabs(f_dc_g - f_dc[1]) < 0.01f &&
        std::fabs(f_dc_b - f_dc[2]) < 0.01f,
        "encodeColor round-trip");
  printf("PASS: encodeColor round-trip\n");
}

// ── encodeScale round-trip ────────────────────────────────────────────────

static void testEncodeScaleRoundTrip() {
  // Raw float arrays instead of Eigen::Vector3f to avoid Eigen dependency in test
  float log_scale[3] = {-1.0f, 0.5f, 2.0f};
  float sMin[3] = {0.1f, 0.1f, 0.1f};
  float sMax[3] = {10.0f, 10.0f, 10.0f};

  uint16_t enc[3];
  // The library API uses Eigen; call with const-cast to float arrays.
  // encodeScale signature: (const float*, const Eigen::Vector3f&, const Eigen::Vector3f&, uint16_t*)
  // We avoid Eigen by calling a wrapper below.
  // For the test we just call the low-level function:
  auto sMinE = Eigen::Vector3f(sMin[0], sMin[1], sMin[2]);
  auto sMaxE = Eigen::Vector3f(sMax[0], sMax[1], sMax[2]);
  splat::encodeScale(log_scale, sMinE, sMaxE, enc);

  for (int i = 0; i < 3; ++i) {
    float norm = enc[i] / 65535.0f;
    float linear = sMin[i] + norm * (sMax[i] - sMin[i]);
    float log_back = std::log(linear);
    check(std::fabs(log_back - log_scale[i]) < 0.02f, "encodeScale round-trip");
  }
  printf("PASS: encodeScale round-trip\n");
}

// ── encodeSh round-trip ───────────────────────────────────────────────────

static void testEncodeShTripletRoundTrip() {
  float sh_min = -2.0f, sh_max = 2.0f;
  uint32_t enc = splat::encodeShTriplet(0.5f, -0.3f, 1.2f, sh_min, sh_max);

  float r, g, b;
  decodePacked_11_10_11(r, g, b, enc);

  auto denorm = [=](float v) { return sh_min + v * (sh_max - sh_min); };
  check(std::fabs(denorm(r) - 0.5f) < 0.01f &&
        std::fabs(denorm(g) - (-0.3f)) < 0.01f &&
        std::fabs(denorm(b) - 1.2f) < 0.01f,
        "encodeShTriplet round-trip");
  printf("PASS: encodeShTriplet round-trip\n");
}

static void testEncodeShCoefRoundTrip() {
  float sh_min = -3.0f, sh_max = 3.0f;
  // Fill values well within range
  float f_rest[45];
  for (int i = 0; i < 45; ++i) f_rest[i] = (i - 22.0f) * 0.1f;

  uint32_t out[16];
  splat::encodeShCoef(f_rest, sh_min, sh_max, out);

  auto denorm = [=](float v) { return sh_min + v * (sh_max - sh_min); };
  bool ok = true;
  for (int band = 0; band < 15 && ok; ++band) {
    float r, g, b;
    decodePacked_11_10_11(r, g, b, out[band]);
    ok = std::fabs(denorm(r) - f_rest[band]) < 0.02f &&
         std::fabs(denorm(g) - f_rest[band+15]) < 0.05f &&
         std::fabs(denorm(b) - f_rest[band+30]) < 0.02f;
  }
  check(ok, "encodeShCoef round-trip");
  check(out[15] == 0, "encodeShCoef padding zero");
  printf("PASS: encodeShCoef round-trip\n");
}

// ── encodeSplat full record test ─────────────────────────────────────────

static void testEncodeSplat() {
  float pos[3] = {10.0f, 20.0f, 30.0f};
  float f_dc[3] = {0.1f, -0.1f, 0.2f};
  float opacity = 0.8f;
  float scale_log[3] = {-0.5f, 0.0f, 0.5f};
  float rot[4] = {0.7071f, 0.0f, 0.7071f, 0.0f};
  float f_rest[45] = {};
  auto sMinE = Eigen::Vector3f(0.01f, 0.01f, 0.01f);
  auto sMaxE = Eigen::Vector3f(20.0f, 20.0f, 20.0f);

  uint8_t data[32], sh[64];

  splat::encodeSplat(pos, f_dc, opacity, scale_log, rot, f_rest,
                     sMinE, sMaxE, -3.0f, 3.0f, true, data, sh);

  // Verify position
  float px, py, pz;
  std::memcpy(&px, data, 4);
  std::memcpy(&py, data+4, 4);
  std::memcpy(&pz, data+8, 4);
  check(px == 10.0f && py == 20.0f && pz == 30.0f, "encodeSplat position");

  // Verify color not zero
  uint32_t color;
  std::memcpy(&color, data+12, 4);
  check(color != 0, "encodeSplat color");

  // Verify rotation not zero
  uint32_t rot_enc;
  std::memcpy(&rot_enc, data+22, 4);
  check(rot_enc != 0, "encodeSplat rotation");

  printf("PASS: encodeSplat full\n");
}

// ── Main ──────────────────────────────────────────────────────────────────

int main() {
  printf("=== LCC Writer Tests ===\n\n");

  printf("--- encodeRotation ---\n");
  testEncodeRotationIdentity();
  testEncodeRotationArbitrary();
  testEncodeRotationAxes();

  printf("\n--- encodeColor ---\n");
  testEncodeColorGray();
  testEncodeColorRoundTrip();

  printf("\n--- encodeScale ---\n");
  testEncodeScaleRoundTrip();

  printf("\n--- encodeSh ---\n");
  testEncodeShTripletRoundTrip();
  testEncodeShCoefRoundTrip();

  printf("\n--- encodeSplat ---\n");
  testEncodeSplat();

  printf("\n=== All tests passed ===\n");
  return 0;
}
