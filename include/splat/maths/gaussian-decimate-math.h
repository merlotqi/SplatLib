/**
 * @file splat/maths/gaussian-decimate-math.h
 * @brief Numerics for Gaussian splat decimation
 *
 * Provides inline math utilities for splat decimation including covariance
 * computation, Jacobi eigendecomposition, quaternion conversions, and
 * radix sort for float-keyed indices.
 *
 * References:
 * - [NanoGS simplification](https://arxiv.org/abs/2411.10223)
 * - [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
 *
 * @namespace splat::decimate_math
 * @brief Inline numerics for Gaussian splat decimation algorithms.
 *        Contains PRNG, covariance math, eigendecomposition, and sorting utilities.
 */
 
#pragma once

#include <splat/maths/maths.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace splat {

/**
 * @namespace decimate_math
 * @brief Inline numerics for Gaussian splat decimation.
 *
 * Contains PRNG, covariance math, eigendecomposition, quaternion conversions,
 * and radix sort utilities used in splat decimation algorithms.
 */
namespace decimate_math {

/** @{ */
/** @name Mathematical Constants */

/** @brief Precomputed (2*pi)^(3/2) constant for Gaussian PDF. */
inline constexpr double kTwoPiPow1_5 = 15.749601955654894;
/** @brief Precomputed log(2*pi) constant. */
inline constexpr double kLog2Pi = 1.8378770664093453;
/** @brief Small epsilon for diagonal covariance regularization. */
inline constexpr double kEpsCov = 1e-8;
/** @brief Pi constant (avoids dependency on M_PI). */
inline constexpr double kPi = 3.14159265358979323846;
/** @} */

/**
 * @brief Compute log-odds (logit) transformation
 *
 * Applies logit(p) = log(p / (1-p)) with input clamping to avoid
 * numerical overflow at p=0 or p=1.
 *
 * @param p Probability value, clamped to (1e-7, 1-1e-7)
 * @return Log-odds value in (-inf, +inf)
 */
inline double logit(double p) {
  p = std::max(1e-7, std::min(1.0 - 1e-7, p));
  return std::log(p / (1.0 - p));
}

/**
 * @brief Numerically stable log-sum-exp for two values
 *
 * Computes log(exp(a) + exp(b)) using the log-sum-exp trick to avoid
 * overflow when a or b are large.
 *
 * @param a First value
 * @param b Second value
 * @return log(exp(a) + exp(b))
 */
inline double log_add_exp(double a, double b) {
  if (a == -std::numeric_limits<double>::infinity()) return b;
  if (b == -std::numeric_limits<double>::infinity()) return a;
  const double m = std::max(a, b);
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

/**
 * @brief 32-bit signed multiply returning low 32 bits
 *
 * Emulates JavaScript Math.imul() semantics for deterministic
 * cross-platform PRNG behavior.
 *
 * @param a First operand
 * @param b Second operand
 * @return Low 32 bits of signed multiplication
 */
inline uint32_t imul_u32(uint32_t a, uint32_t b) {
  return static_cast<uint32_t>(static_cast<int64_t>(static_cast<int32_t>(a)) * static_cast<int32_t>(b));
}

/**
 * @brief Mulberry32 pseudo-random number generator
 *
 * A fast, deterministic 32-bit PRNG suitable for reproducible
 * sampling in decimation algorithms.
 */
class Mulberry32 {
  uint32_t seed_;

 public:
  /**
   * @brief Construct with initial seed
   * @param seed Initial PRNG state
   */
  explicit Mulberry32(uint32_t seed) : seed_(seed) {}

  /**
   * @brief Generate next uniform random value in [0, 1)
   * @return Double-precision random value
   */
  double next01() {
    seed_ += 0x6d2b79f5u;
    uint32_t t = seed_;
    t = imul_u32(t ^ (t >> 15), t | 1u);
    t ^= t + imul_u32(t ^ (t >> 7), t | 61u);
    return static_cast<double>(t ^ (t >> 14)) / 4294967296.0;
  }
};

/**
 * @brief Generate 3D Gaussian samples using Box-Muller transform
 *
 * Produces n samples from a standard normal distribution using the
 * Box-Muller method with the Mulberry32 PRNG.
 *
 * @param n Number of samples to generate
 * @param seed PRNG seed for reproducibility
 * @return Vector of 3D sample points
 */
inline std::vector<std::array<double, 3>> make_gaussian_samples(int n, uint32_t seed) {
  Mulberry32 rand(seed);
  std::vector<std::array<double, 3>> out;
  out.reserve(static_cast<size_t>(n));
  while (static_cast<int>(out.size()) < n) {
    double u1 = std::max(rand.next01(), 1e-12);
    double u2 = rand.next01();
    double u3 = std::max(rand.next01(), 1e-12);
    double u4 = rand.next01();
    const double r1 = std::sqrt(-2.0 * std::log(u1));
    const double t1 = 2.0 * kPi * u2;
    const double r2 = std::sqrt(-2.0 * std::log(u3));
    const double t2 = 2.0 * kPi * u4;
    out.push_back({r1 * std::cos(t1), r1 * std::sin(t1), r2 * std::cos(t2)});
  }
  return out;
}

/**
 * @brief Convert unit quaternion to 3x3 rotation matrix
 *
 * Computes the rotation matrix from a unit quaternion (qw, qx, qy, qz).
 * Output is stored in row-major order at out[o..o+8].
 *
 * @param qw Scalar (w) component of quaternion
 * @param qx X component of quaternion
 * @param qy Y component of quaternion
 * @param qz Z component of quaternion
 * @param out Output array for 3x3 matrix (row-major)
 * @param o Offset in output array
 */
inline void quat_to_rotmat(double qw, double qx, double qy, double qz, double* out, size_t o) {
  const double ww = qw * qw, xx = qx * qx, yy = qy * qy, zz = qz * qz;
  const double wx = qw * qx, wy = qw * qy, wz = qw * qz;
  const double xy = qx * qy, xz = qx * qz, yz = qy * qz;
  out[o + 0] = 1 - 2 * (yy + zz);
  out[o + 1] = 2 * (xy - wz);
  out[o + 2] = 2 * (xz + wy);
  out[o + 3] = 2 * (xy + wz);
  out[o + 4] = 1 - 2 * (xx + zz);
  out[o + 5] = 2 * (yz - wx);
  out[o + 6] = 2 * (xz - wy);
  out[o + 7] = 2 * (yz + wx);
  out[o + 8] = 1 - 2 * (xx + yy);
}

/**
 * @brief Transpose a 3x3 matrix stored in row-major order
 *
 * @param src Source matrix (9 elements, row-major)
 * @param so Offset in source array
 * @param dst Destination matrix (9 elements, row-major)
 * @param doff Offset in destination array
 */
inline void transpose3(const double* src, size_t so, double* dst, size_t doff) {
  dst[doff + 0] = src[so + 0];
  dst[doff + 1] = src[so + 3];
  dst[doff + 2] = src[so + 6];
  dst[doff + 3] = src[so + 1];
  dst[doff + 4] = src[so + 4];
  dst[doff + 5] = src[so + 7];
  dst[doff + 6] = src[so + 2];
  dst[doff + 7] = src[so + 5];
  dst[doff + 8] = src[so + 8];
}

/**
 * @brief Compute covariance matrix from rotation and variances
 *
 * Computes Sigma = R * diag(vx, vy, vz) * R^T, stored in row-major order.
 * The result is symmetric.
 *
 * @param R Rotation matrix (9 elements, row-major)
 * @param r Offset in R array
 * @param vx Variance along local X axis
 * @param vy Variance along local Y axis
 * @param vz Variance along local Z axis
 * @param out Output covariance matrix (9 elements, row-major)
 * @param o Offset in output array
 */
inline void sigma_from_rot_var(const double* R, size_t r, double vx, double vy, double vz, double* out, size_t o) {
  const double r00 = R[r], r01 = R[r + 1], r02 = R[r + 2];
  const double r10 = R[r + 3], r11 = R[r + 4], r12 = R[r + 5];
  const double r20 = R[r + 6], r21 = R[r + 7], r22 = R[r + 8];
  out[o + 0] = r00 * r00 * vx + r01 * r01 * vy + r02 * r02 * vz;
  out[o + 1] = r00 * r10 * vx + r01 * r11 * vy + r02 * r12 * vz;
  out[o + 2] = r00 * r20 * vx + r01 * r21 * vy + r02 * r22 * vz;
  out[o + 3] = out[o + 1];
  out[o + 4] = r10 * r10 * vx + r11 * r11 * vy + r12 * r12 * vz;
  out[o + 5] = r10 * r20 * vx + r11 * r21 * vy + r12 * r22 * vz;
  out[o + 6] = out[o + 2];
  out[o + 7] = out[o + 5];
  out[o + 8] = r20 * r20 * vx + r21 * r21 * vy + r22 * r22 * vz;
}

/**
 * @brief Compute determinant of a 3x3 matrix
 *
 * @param A Matrix array (9 elements, row-major)
 * @param o Offset in array
 * @return Determinant value
 */
inline double det3(const double* A, size_t o) {
  return A[o] * (A[o + 4] * A[o + 8] - A[o + 5] * A[o + 7]) -
         A[o + 1] * (A[o + 3] * A[o + 8] - A[o + 5] * A[o + 6]) +
         A[o + 2] * (A[o + 3] * A[o + 7] - A[o + 4] * A[o + 6]);
}

/**
 * @brief Evaluate log PDF of Gaussian with diagonal-covariance in eigen-frame
 *
 * Computes log N(x | mu, R*diag(v)*R^T) efficiently by transforming
 * to the eigen-frame and using precomputed inverse variances.
 *
 * @param x,y,z Query point coordinates
 * @param mx,my,mz Gaussian mean coordinates
 * @param R Rotation matrix (eigenframe, row-major)
 * @param ro Offset in R array
 * @param invx,invy,invz Inverse eigenvalues (1/variance)
 * @param logdet Log-determinant of covariance matrix
 * @return Log probability density value
 */
inline double gauss_logpdf_diagrot(double x, double y, double z, double mx, double my, double mz, const double* R,
                                   size_t ro, double invx, double invy, double invz, double logdet) {
  const double dx = x - mx, dy = y - my, dz = z - mz;
  const double y0 = dx * R[ro] + dy * R[ro + 3] + dz * R[ro + 6];
  const double y1 = dx * R[ro + 1] + dy * R[ro + 4] + dz * R[ro + 7];
  const double y2 = dx * R[ro + 2] + dy * R[ro + 5] + dz * R[ro + 8];
  const double quad = y0 * y0 * invx + y1 * y1 * invy + y2 * y2 * invz;
  return -0.5 * (3 * kLog2Pi + logdet + quad);
}

/**
 * @brief Eigenvalue decomposition result for 3x3 symmetric matrix
 *
 * Stores eigenvalues and corresponding eigenvectors (as columns
 * in a row-major matrix).
 */
struct Symmetric3x3Eigen {
  double values[3];   ///< Eigenvalues
  double vectors[9];  ///< Eigenvectors as columns (row-major storage)
};

/**
 * @brief Eigendecompose a 3x3 symmetric matrix using Jacobi iteration
 *
 * Iteratively diagonalizes the input symmetric matrix to extract
 * eigenvalues and eigenvectors. Converges within 24 sweeps for
 * well-conditioned matrices.
 *
 * @param Ain Input symmetric matrix (9 elements, row-major)
 * @return Symmetric3x3Eigen with eigenvalues and eigenvector matrix
 */
inline Symmetric3x3Eigen eigen_symmetric_3x3(const double* Ain) {
  double A[9];
  std::memcpy(A, Ain, sizeof(A));
  double V[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  for (int iter = 0; iter < 24; ++iter) {
    int p = 0, q = 1;
    double max_abs = std::abs(A[1]);
    if (std::abs(A[2]) > max_abs) {
      p = 0;
      q = 2;
      max_abs = std::abs(A[2]);
    }
    if (std::abs(A[5]) > max_abs) {
      p = 1;
      q = 2;
      max_abs = std::abs(A[5]);
    }
    if (max_abs < 1e-12) break;

    const int pp = 3 * p + p, qq = 3 * q + q, pq = 3 * p + q;
    const double app = A[pp], aqq = A[qq], apq = A[pq];
    const double tau = (aqq - app) / (2 * apq);
    const double t = (tau >= 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1 + tau * tau));
    const double c = 1 / std::sqrt(1 + t * t);
    const double s = t * c;

    for (int k = 0; k < 3; ++k) {
      if (k == p || k == q) continue;
      const int kp = 3 * k + p, kq = 3 * k + q;
      const int pk = 3 * p + k, qk = 3 * q + k;
      const double akp = A[kp], akq = A[kq];
      A[kp] = c * akp - s * akq;
      A[pk] = A[kp];
      A[kq] = s * akp + c * akq;
      A[qk] = A[kq];
    }
    A[pp] = c * c * app - 2 * s * c * apq + s * s * aqq;
    A[qq] = s * s * app + 2 * s * c * apq + c * c * aqq;
    A[pq] = 0;
    A[3 * q + p] = 0;

    for (int k = 0; k < 3; ++k) {
      const int kp = 3 * k + p, kq = 3 * k + q;
      const double vkp = V[kp], vkq = V[kq];
      V[kp] = c * vkp - s * vkq;
      V[kq] = s * vkp + c * vkq;
    }
  }

  Symmetric3x3Eigen ev{};
  ev.values[0] = A[0];
  ev.values[1] = A[4];
  ev.values[2] = A[8];
  std::memcpy(ev.vectors, V, sizeof(V));
  return ev;
}

/**
 * @brief Convert 3x3 rotation matrix to unit quaternion
 *
 * Extracts quaternion (w,x,y,z) from a rotation matrix using the
 * branch-stable method based on the largest diagonal/tr element.
 * Output is normalized to unit length.
 *
 * @param R Input rotation matrix (9 elements, row-major)
 * @param o Offset in R array
 * @param q_out Output quaternion (4 elements: w,x,y,z)
 */
inline void rotmat_to_quat(const double* R, size_t o, double* q_out) {
  const double m00 = R[o], m11 = R[o + 4], m22 = R[o + 8];
  const double tr = m00 + m11 + m22;
  double qw, qx, qy, qz;

  if (tr > 0) {
    const double S = std::sqrt(tr + 1) * 2;
    qw = 0.25 * S;
    qx = (R[o + 7] - R[o + 5]) / S;
    qy = (R[o + 2] - R[o + 6]) / S;
    qz = (R[o + 3] - R[o + 1]) / S;
  } else if (m00 > m11 && m00 > m22) {
    const double S = std::sqrt(1 + m00 - m11 - m22) * 2;
    qw = (R[o + 7] - R[o + 5]) / S;
    qx = 0.25 * S;
    qy = (R[o + 1] + R[o + 3]) / S;
    qz = (R[o + 2] + R[o + 6]) / S;
  } else if (m11 > m22) {
    const double S = std::sqrt(1 + m11 - m00 - m22) * 2;
    qw = (R[o + 2] - R[o + 6]) / S;
    qx = (R[o + 1] + R[o + 3]) / S;
    qy = 0.25 * S;
    qz = (R[o + 5] + R[o + 7]) / S;
  } else {
    const double S = std::sqrt(1 + m22 - m00 - m11) * 2;
    qw = (R[o + 3] - R[o + 1]) / S;
    qx = (R[o + 2] + R[o + 6]) / S;
    qy = (R[o + 5] + R[o + 7]) / S;
    qz = 0.25 * S;
  }

  const double n = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  const double inv = 1 / std::max(n, 1e-12);
  q_out[0] = qw * inv;
  q_out[1] = qx * inv;
  q_out[2] = qy * inv;
  q_out[3] = qz * inv;
}

/**
 * @brief LSD radix sort for indices keyed by float values
 *
 * Sorts indices by ascending float key values using 4-pass LSD
 * radix sort. Non-finite keys (NaN/Inf) are skipped. Handles
 * negative floats correctly via sign-bit manipulation.
 *
 * @param out Output index array (must have capacity >= count)
 * @param count Number of elements to sort
 * @param keys Float key values for each element
 * @return Number of valid (finite) elements sorted
 */
inline size_t radix_sort_indices_by_float(uint32_t* out, size_t count, const float* keys) {
  static_assert(sizeof(float) == sizeof(uint32_t), "float size");
  const auto* key_bits = reinterpret_cast<const uint32_t*>(keys);

  std::vector<uint32_t> sort_keys(count);
  size_t valid_count = 0;
  for (size_t i = 0; i < count; ++i) {
    const uint32_t bits = key_bits[i];
    if ((bits & 0x7F800000u) == 0x7F800000u) continue;
    sort_keys[valid_count] = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    out[valid_count] = static_cast<uint32_t>(i);
    ++valid_count;
  }

  if (valid_count <= 1) return valid_count;

  const size_t n = valid_count;
  std::vector<uint32_t> temp(n);
  std::vector<uint32_t> temp_keys(n);
  std::vector<uint32_t> counts(256);

  for (int pass = 0; pass < 4; ++pass) {
    const unsigned shift = static_cast<unsigned>(pass << 3);
    const uint32_t* src_idx = (pass & 1) ? temp.data() : out;
    uint32_t* dst_idx = (pass & 1) ? out : temp.data();
    const uint32_t* src_k = (pass & 1) ? temp_keys.data() : sort_keys.data();
    uint32_t* dst_k = (pass & 1) ? sort_keys.data() : temp_keys.data();

    std::fill(counts.begin(), counts.end(), 0u);
    for (size_t i = 0; i < n; ++i) {
      counts[(src_k[i] >> shift) & 0xFFu]++;
    }

    uint32_t sum = 0;
    for (int b = 0; b < 256; ++b) {
      const uint32_t c = counts[static_cast<size_t>(b)];
      counts[static_cast<size_t>(b)] = sum;
      sum += c;
    }

    for (size_t i = 0; i < n; ++i) {
      const unsigned bucket = (src_k[i] >> shift) & 0xFFu;
      const size_t pos = counts[bucket]++;
      dst_idx[pos] = src_idx[i];
      dst_k[pos] = src_k[i];
    }
  }

  return valid_count;
}

}  // namespace decimate_math
}  // namespace splat
