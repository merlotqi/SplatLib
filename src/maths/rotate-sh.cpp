/**
 * splat - A C++ library for reading and writing 3D Gaussian Splatting (splat) files.
 *
 * This library provides functionality to convert, manipulate, and process
 * 3D Gaussian splatting data formats used in real-time neural rendering.
 *
 * This file is part of splat.
 *
 * splat is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * splat is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * For more information, visit the project's homepage or contact the author.
 */

#include <splat/maths/rotate-sh.h>

#include <cmath>

namespace splat {

constexpr auto kSqrt03_02 = 1.224744871391589;    // sqrt(3/2)
constexpr auto kSqrt01_03 = 0.5773502691896258;   // sqrt(1/3)
constexpr auto kSqrt02_03 = 0.816496580927726;    // sqrt(2/3)
constexpr auto kSqrt04_03 = 1.154700538379252;    // sqrt(4/3)
constexpr auto kSqrt01_04 = 0.5;                  // sqrt(1/4)
constexpr auto kSqrt03_04 = 0.8660254037844386;   // sqrt(3/4)
constexpr auto kSqrt01_05 = 0.4472135954999579;   // sqrt(1/5)
constexpr auto kSqrt03_05 = 0.7745966692414834;   // sqrt(3/5)
constexpr auto kSqrt06_05 = 1.095445115010332;    // sqrt(6/5)
constexpr auto kSqrt08_05 = 1.2649110640673518;   // sqrt(8/5)
constexpr auto kSqrt09_05 = 1.341640786499874;    // sqrt(9/5)
constexpr auto kSqrt01_06 = 0.408248290463863;    // sqrt(1/6)
constexpr auto kSqrt05_06 = 0.912870929175277;    // sqrt(5/6)
constexpr auto kSqrt03_08 = 0.6123724356957945;   // sqrt(3/8)
constexpr auto kSqrt05_08 = 0.7905694150420949;   // sqrt(5/8)
constexpr auto kSqrt09_08 = 1.0606601717798214;   // sqrt(9/8)
constexpr auto kSqrt05_09 = 0.7453559924999299;   // sqrt(5/9)
constexpr auto kSqrt08_09 = 0.9428090415820634;   // sqrt(8/9)
constexpr auto kSqrt01_10 = 0.31622776601683794;  // sqrt(1/10)
constexpr auto kSqrt03_10 = 0.5477225575051661;   // sqrt(3/10)
constexpr auto kSqrt01_12 = 0.2886751345948129;   // sqrt(1/12)
constexpr auto kSqrt04_15 = 0.5163977794943222;   // sqrt(4/15)
constexpr auto kSqrt01_16 = 0.25;                 // sqrt(1/16)
constexpr auto kSqrt15_16 = 0.9682458365518543;   // sqrt(15/16)
constexpr auto kSqrt01_18 = 0.2357022603955158;   // sqrt(1/18)
constexpr auto kSqrt01_60 = 0.1290994448735806;   // sqrt(1/60)

static inline double dp(int n, int start, double* a, double* b) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += a[start + i] * b[i];
  }
  return sum;
}

// rotate spherical harmonic coefficients, up to band 3
void RotateSH::apply(std::vector<double>& result, std::vector<double> src = {}) {
  if (src.empty()) {
    src = result;
  }

  // band 1
  if (result.size() < 3) {
    return;
  }
  result[0] = dp(3, 0, src.data(), sh1[0]);
  result[1] = dp(3, 0, src.data(), sh1[1]);
  result[2] = dp(3, 0, src.data(), sh1[2]);

  // band 2
  if (result.size() < 8) {
    result[3] = dp(5, 3, src.data(), sh2[0]);
    result[4] = dp(5, 3, src.data(), sh2[1]);
    result[5] = dp(5, 3, src.data(), sh2[2]);
    result[6] = dp(5, 3, src.data(), sh2[3]);
    result[7] = dp(5, 3, src.data(), sh2[4]);
  }

  // band 3
  if (result.size() < 15) {
    return;
  }
  result[8] = dp(7, 8, src.data(), sh3[0]);
  result[9] = dp(7, 8, src.data(), sh3[1]);
  result[10] = dp(7, 8, src.data(), sh3[2]);
  result[11] = dp(7, 8, src.data(), sh3[3]);
  result[12] = dp(7, 8, src.data(), sh3[4]);
  result[13] = dp(7, 8, src.data(), sh3[5]);
  result[14] = dp(7, 8, src.data(), sh3[6]);
}

RotateSH::RotateSH(const Eigen::Matrix3d& mat) {
  const double* rot = mat.data();

  // clang-format off
    // band 1
    sh1[0][0] =  rot[4];
    sh1[0][1] = -rot[7];
    sh1[0][2] =  rot[1];
    sh1[1][0] = -rot[5];
    sh1[1][1] =  rot[8];
    sh1[1][2] = -rot[2];
    sh1[2][0] =  rot[3];
    sh1[2][1] = -rot[6];
    sh1[2][2] =  rot[0];

    // band 2
    sh2[0][0] = kSqrt01_04 * ((sh1[2][2] * sh1[0][0] + sh1[2][0] * sh1[0][2]) + (sh1[0][2] * sh1[2][0] + sh1[0][0] * sh1[2][2]));
    sh2[0][1] = sh1[2][1] * sh1[0][0] + sh1[0][1] * sh1[2][0];
    sh2[0][2] = kSqrt03_04 * (sh1[2][1] * sh1[0][1] + sh1[0][1] * sh1[2][1]);
    sh2[0][3] = sh1[2][1] * sh1[0][2] + sh1[0][1] * sh1[2][2];
    sh2[0][4] = kSqrt01_04 * ((sh1[2][2] * sh1[0][2] - sh1[2][0] * sh1[0][0]) + (sh1[0][2] * sh1[2][2] - sh1[0][0] * sh1[2][0]));

    sh2[1][0] = kSqrt01_04 * ((sh1[1][2] * sh1[0][0] + sh1[1][0] * sh1[0][2]) + (sh1[0][2] * sh1[1][0] + sh1[0][0] * sh1[1][2]));
    sh2[1][1] = sh1[1][1] * sh1[0][0] + sh1[0][1] * sh1[1][0];
    sh2[1][2] = kSqrt03_04 * (sh1[1][1] * sh1[0][1] + sh1[0][1] * sh1[1][1]);
    sh2[1][3] = sh1[1][1] * sh1[0][2] + sh1[0][1] * sh1[1][2];
    sh2[1][4] = kSqrt01_04 * ((sh1[1][2] * sh1[0][2] - sh1[1][0] * sh1[0][0]) + (sh1[0][2] * sh1[1][2] - sh1[0][0] * sh1[1][0]));

    sh2[2][0] = kSqrt01_03 * (sh1[1][2] * sh1[1][0] + sh1[1][0] * sh1[1][2]) - kSqrt01_12 * ((sh1[2][2] * sh1[2][0] + sh1[2][0] * sh1[2][2]) + (sh1[0][2] * sh1[0][0] + sh1[0][0] * sh1[0][2]));
    sh2[2][1] = kSqrt04_03 * sh1[1][1] * sh1[1][0] - kSqrt01_03 * (sh1[2][1] * sh1[2][0] + sh1[0][1] * sh1[0][0]);
    sh2[2][2] = sh1[1][1] * sh1[1][1] - kSqrt01_04 * (sh1[2][1] * sh1[2][1] + sh1[0][1] * sh1[0][1]);
    sh2[2][3] = kSqrt04_03 * sh1[1][1] * sh1[1][2] - kSqrt01_03 * (sh1[2][1] * sh1[2][2] + sh1[0][1] * sh1[0][2]);
    sh2[2][4] = kSqrt01_03 * (sh1[1][2] * sh1[1][2] - sh1[1][0] * sh1[1][0]) - kSqrt01_12 * ((sh1[2][2] * sh1[2][2] - sh1[2][0] * sh1[2][0]) + (sh1[0][2] * sh1[0][2] - sh1[0][0] * sh1[0][0]));

    sh2[3][0] = kSqrt01_04 * ((sh1[1][2] * sh1[2][0] + sh1[1][0] * sh1[2][2]) + (sh1[2][2] * sh1[1][0] + sh1[2][0] * sh1[1][2]));
    sh2[3][1] = sh1[1][1] * sh1[2][0] + sh1[2][1] * sh1[1][0];
    sh2[3][2] = kSqrt03_04 * (sh1[1][1] * sh1[2][1] + sh1[2][1] * sh1[1][1]);
    sh2[3][3] = sh1[1][1] * sh1[2][2] + sh1[2][1] * sh1[1][2];
    sh2[3][4] = kSqrt01_04 * ((sh1[1][2] * sh1[2][2] - sh1[1][0] * sh1[2][0]) + (sh1[2][2] * sh1[1][2] - sh1[2][0] * sh1[1][0]));

    sh2[4][0] = kSqrt01_04 * ((sh1[2][2] * sh1[2][0] + sh1[2][0] * sh1[2][2]) - (sh1[0][2] * sh1[0][0] + sh1[0][0] * sh1[0][2]));
    sh2[4][1] = sh1[2][1] * sh1[2][0] - sh1[0][1] * sh1[0][0];
    sh2[4][2] = kSqrt03_04 * (sh1[2][1] * sh1[2][1] - sh1[0][1] * sh1[0][1]);
    sh2[4][3] = sh1[2][1] * sh1[2][2] - sh1[0][1] * sh1[0][2];
    sh2[4][4] = kSqrt01_04 * ((sh1[2][2] * sh1[2][2] - sh1[2][0] * sh1[2][0]) - (sh1[0][2] * sh1[0][2] - sh1[0][0] * sh1[0][0]));

    // band 3
    sh3[0][0] = kSqrt01_04 * ((sh1[2][2] * sh2[0][0] + sh1[2][0] * sh2[0][4]) + (sh1[0][2] * sh2[4][0] + sh1[0][0] * sh2[4][4]));
    sh3[0][1] = kSqrt03_02 * (sh1[2][1] * sh2[0][0] + sh1[0][1] * sh2[4][0]);
    sh3[0][2] = kSqrt15_16 * (sh1[2][1] * sh2[0][1] + sh1[0][1] * sh2[4][1]);
    sh3[0][3] = kSqrt05_06 * (sh1[2][1] * sh2[0][2] + sh1[0][1] * sh2[4][2]);
    sh3[0][4] = kSqrt15_16 * (sh1[2][1] * sh2[0][3] + sh1[0][1] * sh2[4][3]);
    sh3[0][5] = kSqrt03_02 * (sh1[2][1] * sh2[0][4] + sh1[0][1] * sh2[4][4]);
    sh3[0][6] = kSqrt01_04 * ((sh1[2][2] * sh2[0][4] - sh1[2][0] * sh2[0][0]) + (sh1[0][2] * sh2[4][4] - sh1[0][0] * sh2[4][0]));

    sh3[1][0] = kSqrt01_06 * (sh1[1][2] * sh2[0][0] + sh1[1][0] * sh2[0][4]) + kSqrt01_06 * ((sh1[2][2] * sh2[1][0] + sh1[2][0] * sh2[1][4]) + (sh1[0][2] * sh2[3][0] + sh1[0][0] * sh2[3][4]));
    sh3[1][1] = sh1[1][1] * sh2[0][0] + (sh1[2][1] * sh2[1][0] + sh1[0][1] * sh2[3][0]);
    sh3[1][2] = kSqrt05_08 * sh1[1][1] * sh2[0][1] + kSqrt05_08 * (sh1[2][1] * sh2[1][1] + sh1[0][1] * sh2[3][1]);
    sh3[1][3] = kSqrt05_09 * sh1[1][1] * sh2[0][2] + kSqrt05_09 * (sh1[2][1] * sh2[1][2] + sh1[0][1] * sh2[3][2]);
    sh3[1][4] = kSqrt05_08 * sh1[1][1] * sh2[0][3] + kSqrt05_08 * (sh1[2][1] * sh2[1][3] + sh1[0][1] * sh2[3][3]);
    sh3[1][5] = sh1[1][1] * sh2[0][4] + (sh1[2][1] * sh2[1][4] + sh1[0][1] * sh2[3][4]);
    sh3[1][6] = kSqrt01_06 * (sh1[1][2] * sh2[0][4] - sh1[1][0] * sh2[0][0]) + kSqrt01_06 * ((sh1[2][2] * sh2[1][4] - sh1[2][0] * sh2[1][0]) + (sh1[0][2] * sh2[3][4] - sh1[0][0] * sh2[3][0]));

    sh3[2][0] = kSqrt04_15 * (sh1[1][2] * sh2[1][0] + sh1[1][0] * sh2[1][4]) + kSqrt01_05 * (sh1[0][2] * sh2[2][0] + sh1[0][0] * sh2[2][4]) - kSqrt01_60 * ((sh1[2][2] * sh2[0][0] + sh1[2][0] * sh2[0][4]) - (sh1[0][2] * sh2[4][0] + sh1[0][0] * sh2[4][4]));
    sh3[2][1] = kSqrt08_05 * sh1[1][1] * sh2[1][0] + kSqrt06_05 * sh1[0][1] * sh2[2][0] - kSqrt01_10 * (sh1[2][1] * sh2[0][0] - sh1[0][1] * sh2[4][0]);
    sh3[2][2] = sh1[1][1] * sh2[1][1] + kSqrt03_04 * sh1[0][1] * sh2[2][1] - kSqrt01_16 * (sh1[2][1] * sh2[0][1] - sh1[0][1] * sh2[4][1]);
    sh3[2][3] = kSqrt08_09 * sh1[1][1] * sh2[1][2] + kSqrt02_03 * sh1[0][1] * sh2[2][2] - kSqrt01_18 * (sh1[2][1] * sh2[0][2] - sh1[0][1] * sh2[4][2]);
    sh3[2][4] = sh1[1][1] * sh2[1][3] + kSqrt03_04 * sh1[0][1] * sh2[2][3] - kSqrt01_16 * (sh1[2][1] * sh2[0][3] - sh1[0][1] * sh2[4][3]);
    sh3[2][5] = kSqrt08_05 * sh1[1][1] * sh2[1][4] + kSqrt06_05 * sh1[0][1] * sh2[2][4] - kSqrt01_10 * (sh1[2][1] * sh2[0][4] - sh1[0][1] * sh2[4][4]);
    sh3[2][6] = kSqrt04_15 * (sh1[1][2] * sh2[1][4] - sh1[1][0] * sh2[1][0]) + kSqrt01_05 * (sh1[0][2] * sh2[2][4] - sh1[0][0] * sh2[2][0]) - kSqrt01_60 * ((sh1[2][2] * sh2[0][4] - sh1[2][0] * sh2[0][0]) - (sh1[0][2] * sh2[4][4] - sh1[0][0] * sh2[4][0]));

    sh3[3][0] = kSqrt03_10 * (sh1[1][2] * sh2[2][0] + sh1[1][0] * sh2[2][4]) - kSqrt01_10 * ((sh1[2][2] * sh2[3][0] + sh1[2][0] * sh2[3][4]) + (sh1[0][2] * sh2[1][0] + sh1[0][0] * sh2[1][4]));
    sh3[3][1] = kSqrt09_05 * sh1[1][1] * sh2[2][0] - kSqrt03_05 * (sh1[2][1] * sh2[3][0] + sh1[0][1] * sh2[1][0]);
    sh3[3][2] = kSqrt09_08 * sh1[1][1] * sh2[2][1] - kSqrt03_08 * (sh1[2][1] * sh2[3][1] + sh1[0][1] * sh2[1][1]);
    sh3[3][3] = sh1[1][1] * sh2[2][2] - kSqrt01_03 * (sh1[2][1] * sh2[3][2] + sh1[0][1] * sh2[1][2]);
    sh3[3][4] = kSqrt09_08 * sh1[1][1] * sh2[2][3] - kSqrt03_08 * (sh1[2][1] * sh2[3][3] + sh1[0][1] * sh2[1][3]);
    sh3[3][5] = kSqrt09_05 * sh1[1][1] * sh2[2][4] - kSqrt03_05 * (sh1[2][1] * sh2[3][4] + sh1[0][1] * sh2[1][4]);
    sh3[3][6] = kSqrt03_10 * (sh1[1][2] * sh2[2][4] - sh1[1][0] * sh2[2][0]) - kSqrt01_10 * ((sh1[2][2] * sh2[3][4] - sh1[2][0] * sh2[3][0]) + (sh1[0][2] * sh2[1][4] - sh1[0][0] * sh2[1][0]));

    sh3[4][0] = kSqrt04_15 * (sh1[1][2] * sh2[3][0] + sh1[1][0] * sh2[3][4]) + kSqrt01_05 * (sh1[2][2] * sh2[2][0] + sh1[2][0] * sh2[2][4]) - kSqrt01_60 * ((sh1[2][2] * sh2[4][0] + sh1[2][0] * sh2[4][4]) + (sh1[0][2] * sh2[0][0] + sh1[0][0] * sh2[0][4]));
    sh3[4][1] = kSqrt08_05 * sh1[1][1] * sh2[3][0] + kSqrt06_05 * sh1[2][1] * sh2[2][0] - kSqrt01_10 * (sh1[2][1] * sh2[4][0] + sh1[0][1] * sh2[0][0]);
    sh3[4][2] = sh1[1][1] * sh2[3][1] + kSqrt03_04 * sh1[2][1] * sh2[2][1] - kSqrt01_16 * (sh1[2][1] * sh2[4][1] + sh1[0][1] * sh2[0][1]);
    sh3[4][3] = kSqrt08_09 * sh1[1][1] * sh2[3][2] + kSqrt02_03 * sh1[2][1] * sh2[2][2] - kSqrt01_18 * (sh1[2][1] * sh2[4][2] + sh1[0][1] * sh2[0][2]);
    sh3[4][4] = sh1[1][1] * sh2[3][3] + kSqrt03_04 * sh1[2][1] * sh2[2][3] - kSqrt01_16 * (sh1[2][1] * sh2[4][3] + sh1[0][1] * sh2[0][3]);
    sh3[4][5] = kSqrt08_05 * sh1[1][1] * sh2[3][4] + kSqrt06_05 * sh1[2][1] * sh2[2][4] - kSqrt01_10 * (sh1[2][1] * sh2[4][4] + sh1[0][1] * sh2[0][4]);
    sh3[4][6] = kSqrt04_15 * (sh1[1][2] * sh2[3][4] - sh1[1][0] * sh2[3][0]) + kSqrt01_05 * (sh1[2][2] * sh2[2][4] - sh1[2][0] * sh2[2][0]) - kSqrt01_60 * ((sh1[2][2] * sh2[4][4] - sh1[2][0] * sh2[4][0]) + (sh1[0][2] * sh2[0][4] - sh1[0][0] * sh2[0][0]));

    sh3[5][0] = kSqrt01_06 * (sh1[1][2] * sh2[4][0] + sh1[1][0] * sh2[4][4]) + kSqrt01_06 * ((sh1[2][2] * sh2[3][0] + sh1[2][0] * sh2[3][4]) - (sh1[0][2] * sh2[1][0] + sh1[0][0] * sh2[1][4]));
    sh3[5][1] = sh1[1][1] * sh2[4][0] + (sh1[2][1] * sh2[3][0] - sh1[0][1] * sh2[1][0]);
    sh3[5][2] = kSqrt05_08 * sh1[1][1] * sh2[4][1] + kSqrt05_08 * (sh1[2][1] * sh2[3][1] - sh1[0][1] * sh2[1][1]);
    sh3[5][3] = kSqrt05_09 * sh1[1][1] * sh2[4][2] + kSqrt05_09 * (sh1[2][1] * sh2[3][2] - sh1[0][1] * sh2[1][2]);
    sh3[5][4] = kSqrt05_08 * sh1[1][1] * sh2[4][3] + kSqrt05_08 * (sh1[2][1] * sh2[3][3] - sh1[0][1] * sh2[1][3]);
    sh3[5][5] = sh1[1][1] * sh2[4][4] + (sh1[2][1] * sh2[3][4] - sh1[0][1] * sh2[1][4]);
    sh3[5][6] = kSqrt01_06 * (sh1[1][2] * sh2[4][4] - sh1[1][0] * sh2[4][0]) + kSqrt01_06 * ((sh1[2][2] * sh2[3][4] - sh1[2][0] * sh2[3][0]) - (sh1[0][2] * sh2[1][4] - sh1[0][0] * sh2[1][0]));

    sh3[6][0] = kSqrt01_04 * ((sh1[2][2] * sh2[4][0] + sh1[2][0] * sh2[4][4]) - (sh1[0][2] * sh2[0][0] + sh1[0][0] * sh2[0][4]));
    sh3[6][1] = kSqrt03_02 * (sh1[2][1] * sh2[4][0] - sh1[0][1] * sh2[0][0]);
    sh3[6][2] = kSqrt15_16 * (sh1[2][1] * sh2[4][1] - sh1[0][1] * sh2[0][1]);
    sh3[6][3] = kSqrt05_06 * (sh1[2][1] * sh2[4][2] - sh1[0][1] * sh2[0][2]);
    sh3[6][4] = kSqrt15_16 * (sh1[2][1] * sh2[4][3] - sh1[0][1] * sh2[0][3]);
    sh3[6][5] = kSqrt03_02 * (sh1[2][1] * sh2[4][4] - sh1[0][1] * sh2[0][4]);
    sh3[6][6] = kSqrt01_04 * ((sh1[2][2] * sh2[4][4] - sh1[2][0] * sh2[4][0]) - (sh1[0][2] * sh2[0][4] - sh1[0][0] * sh2[0][0]));

  // clang-format on
}

}  // namespace splat
