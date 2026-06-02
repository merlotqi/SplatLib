/**
 * @file splat/maths/coordinate-converter.h
 * @brief Coordinate system conversion for 3DGS position, rotation, and SH data.
 *
 * Supports 16 coordinate systems. Within-family conversions use sign flips.
 * Cross-family conversions use R_x(±π/2) analytic rotation + sign flips.
 */
#pragma once

#include <array>
#include <functional>
#include <cstdint>

namespace splat {

enum class CoordinateSystem : uint32_t {
  UNSPECIFIED = 0,
  LDB = 1,  LUB = 3,  LDF = 5,  LUF = 7,
  RDB = 2,  RUB = 4,  RDF = 6,  RUF = 8,
  LFD = 9,  LFU = 11, LBD = 13, LBU = 15,
  RFD = 10, RFU = 12, RBD = 14, RBU = 16,
  SPZ_DEFAULT = RUB,
  PLY_DEFAULT = RDF,
  GLB_DEFAULT = LUF,
  UNITY_DEFAULT = RUF,
};

struct CoordinateConverter {
  std::array<float, 3> flipP = {1, 1, 1};
  std::array<float, 3> flipQ = {1, 1, 1};
  std::array<float, 24> flipSh;
  std::function<void(float*)> rotFlipPos;
  std::function<void(float*)> rotFlipQuat;
  std::array<std::function<void(float*)>, 4> rotFlipShBands;

  CoordinateConverter();

  void convertPosition(float pos[3]) const;
  void convertRotation(float quat[4]) const;
  void convertSH(float* sh, int numBands, int shDegree) const;
};

CoordinateConverter makeCoordinateConverter(
    CoordinateSystem from, CoordinateSystem to, int shDegree);

}  // namespace splat
