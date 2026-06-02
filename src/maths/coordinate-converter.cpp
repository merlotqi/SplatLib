#include <splat/maths/coordinate-converter.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace splat {

CoordinateConverter::CoordinateConverter() { flipSh.fill(1.0f); }

namespace {

bool needRotation(CoordinateSystem a, CoordinateSystem b) {
  int an = static_cast<int>(a) - 1, bn = static_cast<int>(b) - 1;
  if (an < 0 || bn < 0) return false;
  return ((an >> 3) & 1) != ((bn >> 3) & 1);
}

std::array<bool, 3> axesMatch(CoordinateSystem a, CoordinateSystem b) {
  int an = static_cast<int>(a) - 1, bn = static_cast<int>(b) - 1;
  if (an < 0 || bn < 0) return {true, true, true};
  return {
    ((an >> 0) & 1) == ((bn >> 0) & 1),
    ((an >> 1) & 1) == ((bn >> 1) & 1),
    ((an >> 2) & 1) == ((bn >> 2) & 1)
  };
}

void computeFlips(CoordinateSystem from, CoordinateSystem to,
                  std::array<float, 3>& flipP, std::array<float, 3>& flipQ,
                  std::array<float, 24>& flipSh) {
  auto [xMatch, yMatch, zMatch] = axesMatch(from, to);
  float x = xMatch ? 1.0f : -1.0f;
  float y = yMatch ? 1.0f : -1.0f;
  float z = zMatch ? 1.0f : -1.0f;

  flipP = {x, y, z};
  flipQ = {y * z, x * z, x * y};
  flipSh = {
    y, z, x, x*y, y*z, 1, x*z, 1, y,
    x*y*z, y, z, x, z, x,
    x*y, y*z, x*y, y*z, 1, x*z, 1, x*z, y
  };
}

float kSqrt2() { return std::sqrt(2.0f); }
float kSqrt3() { return std::sqrt(3.0f); }

void rotFlipPosPlus90X(float* p) {
  float y = p[1], z = p[2]; p[1] = -z; p[2] = y;
}
void rotFlipPosMinus90X(float* p) {
  float y = p[1], z = p[2]; p[1] = z; p[2] = -y;
}

void rotFlipQuatPlus90X(float* p) {
  float s = kSqrt2() / 2.0f;
  float w = p[0], x = p[1], y = p[2], z = p[3];
  p[0] = s * (w + x); p[1] = s * (y - z);
  p[2] = s * (y + z); p[3] = s * (w - x);
}
void rotFlipQuatMinus90X(float* p) {
  float s = kSqrt2() / 2.0f;
  float w = p[0], x = p[1], y = p[2], z = p[3];
  p[0] = s * (-w + x); p[1] = s * (y + z);
  p[2] = s * (-y + z); p[3] = s * (w + x);
}

// SH band-0 rotation (l=1: 3 coeffs)
void rotShBand0Plus90X(float* p) { float t0 = p[0]; p[0] = p[1]; p[1] = -t0; }
void rotShBand0Minus90X(float* p) { float t0 = p[0]; p[0] = -p[1]; p[1] = t0; }

// SH band-1 rotation (l=2: 5 coeffs)
void rotShBand1Plus90X(float* p) {
  float s[5]; std::memcpy(s, p, 5 * sizeof(float));
  float s3 = kSqrt3();
  p[0] = s[3]; p[1] = -s[1];
  p[2] = -0.5f * s[2] - (s3 / 2.0f) * s[4];
  p[3] = -s[0];
  p[4] = -(s3 / 2.0f) * s[2] + 0.5f * s[4];
}
void rotShBand1Minus90X(float* p) {
  float s[5]; std::memcpy(s, p, 5 * sizeof(float));
  float s3 = kSqrt3();
  p[0] = -s[3]; p[1] = -s[1];
  p[2] = -0.5f * s[2] - (s3 / 2.0f) * s[4];
  p[3] = s[0];
  p[4] = -(s3 / 2.0f) * s[2] + 0.5f * s[4];
}

}  // namespace

CoordinateConverter makeCoordinateConverter(
    CoordinateSystem from, CoordinateSystem to, int shDegree) {

  CoordinateConverter conv;
  if (from == to || from == CoordinateSystem::UNSPECIFIED ||
      to == CoordinateSystem::UNSPECIFIED) {
    return conv;  // identity
  }

  if (needRotation(from, to)) {
    bool backward = ((static_cast<int>(from) - 1) >> 3) & 1;
    CoordinateSystem innerFrom = backward
        ? static_cast<CoordinateSystem>(static_cast<int>(from) - 8)
        : static_cast<CoordinateSystem>(static_cast<int>(from) + 8);

    computeFlips(innerFrom, to, conv.flipP, conv.flipQ, conv.flipSh);
    auto fp = conv.flipP, fq = conv.flipQ;

    if (backward) {
      conv.rotFlipPos = [fp](float* p) {
        p[0] *= fp[0]; p[1] *= fp[1]; p[2] *= fp[2];
        rotFlipPosMinus90X(p);
      };
      conv.rotFlipQuat = [fq](float* p) {
        p[0] *= fq[0]; p[1] *= fq[1]; p[2] *= fq[2];
        rotFlipQuatMinus90X(p);
      };
      conv.rotFlipShBands[0] = rotShBand0Minus90X;
      conv.rotFlipShBands[1] = rotShBand1Minus90X;
    } else {
      conv.rotFlipPos = [fp](float* p) {
        rotFlipPosPlus90X(p);
        p[0] *= fp[0]; p[1] *= fp[1]; p[2] *= fp[2];
      };
      conv.rotFlipQuat = [fq](float* p) {
        rotFlipQuatPlus90X(p);
        p[0] *= fq[0]; p[1] *= fq[1]; p[2] *= fq[2];
      };
      conv.rotFlipShBands[0] = rotShBand0Plus90X;
      conv.rotFlipShBands[1] = rotShBand1Plus90X;
    }
    conv.flipP = {1, 1, 1};
    conv.flipQ = {1, 1, 1};
    conv.flipSh.fill(1.0f);
  } else {
    computeFlips(from, to, conv.flipP, conv.flipQ, conv.flipSh);
  }
  return conv;
}

void CoordinateConverter::convertPosition(float pos[3]) const {
  if (rotFlipPos) { rotFlipPos(pos); return; }
  for (int i = 0; i < 3; ++i) pos[i] *= flipP[i];
}

void CoordinateConverter::convertRotation(float quat[4]) const {
  if (rotFlipQuat) { rotFlipQuat(quat); return; }
  for (int i = 0; i < 3; ++i) quat[i] *= flipQ[i];
}

void CoordinateConverter::convertSH(float* sh, int numBands, int shDegree) const {
  if (rotFlipShBands[0]) {
    for (int band = 0; band < std::min(shDegree, 2); ++band) {
      int bandStart = band * (band + 2);
      int bandSize = 2 * band + 3;
      if (bandStart + bandSize > numBands) break;
      for (int ch = 0; ch < 3; ++ch) {
        float tmp[9] = {};
        for (int k = 0; k < bandSize; ++k) tmp[k] = sh[(bandStart + k) * 3 + ch];
        if (rotFlipShBands[static_cast<size_t>(band)])
          rotFlipShBands[static_cast<size_t>(band)](tmp);
        for (int k = 0; k < bandSize; ++k) sh[(bandStart + k) * 3 + ch] = tmp[k];
      }
    }
  } else {
    for (int i = 0; i < numBands; ++i) {
      int base = i * 3;
      sh[base + 0] *= flipSh[i];
      sh[base + 1] *= flipSh[i];
      sh[base + 2] *= flipSh[i];
    }
  }
}

}  // namespace splat
