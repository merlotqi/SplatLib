# Decimate Algorithm: Follow Upstream splat-transform v2.1.1 → v2.4.0 (P0–P1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the area-weighted, mass-conserving pairwise merge math from splat-transform upstream (#241) and apply the structural Float64→Float32 + scratch-buffer improvements (#244) to the C++ decimate implementation.

**Architecture:** Changes are localized to `include/splat/maths/gaussian-decimate-math.h` (add `ellipsoid_area`, caller-buffer `eigen_symmetric_3x3` overload) and `src/op/decimate.cpp` (SplatCache f32, MergeScratch, rewrite merge/cost formulas, remove opacity pruning). No new files, no API changes, no new dependencies.

**Tech Stack:** C++17, no new dependencies

**Spec:** `docs/superpowers/specs/2026-05-29-decimate-upstream-follow-design.md`

---

## File Structure

**Modify:**
- `include/splat/maths/gaussian-decimate-math.h` — add `kEllipsoidP` constant, `ellipsoid_area()`, caller-buffer `eigen_symmetric_3x3(Ain, A, V)` overload
- `src/op/decimate.cpp` — `SplatCache` f64→f32 + remove `Rt`, add `MergeScratch`, rewrite `compute_edge_cost`, rewrite `moment_match`, remove opacity pruning

**Do not modify:**
- `include/splat/op/decimate.h` — public API unchanged
- `tests/decimate_test.cpp` — structural assertions still hold

---

### Task 1: Add `ellipsoid_area()` and caller-buffer `eigen_symmetric_3x3()` to math header

**Files:**
- Modify: `include/splat/maths/gaussian-decimate-math.h`

- [ ] **Step 1: Add `kEllipsoidP` constant**

After `kPi` (line 51), insert:

```cpp
/** @brief Knud Thomsen p parameter for ellipsoid surface area approximation. */
inline constexpr double kEllipsoidP = 1.6075;
```

- [ ] **Step 2: Add `ellipsoid_area()` function**

After `log_add_exp()` (line 83), insert:

```cpp
/**
 * @brief Approximate surface area of an ellipsoid with given semi-axes
 *
 * Uses Knud Thomsen's p=1.6075 approximation. Used as the per-splat
 * screen-projection weight in pairwise merge cost and moment-matching,
 * replacing the old volume·α NanoGS weighting.
 *
 * @param sx,sy,sz Semi-axes (scale) along each local axis
 * @return Approximate ellipsoid surface area
 */
inline double ellipsoid_area(double sx, double sy, double sz) {
  const double a = std::pow(sx * sy, kEllipsoidP);
  const double b = std::pow(sx * sz, kEllipsoidP);
  const double c = std::pow(sy * sz, kEllipsoidP);
  return 4 * kPi * std::pow((a + b + c) / 3, 1 / kEllipsoidP);
}
```

- [ ] **Step 3: Add caller-provided-buffer overload of `eigen_symmetric_3x3()`**

After the existing `eigen_symmetric_3x3(const double* Ain)` (after line 347), insert:

```cpp
/**
 * @brief Eigendecompose a 3x3 symmetric matrix using caller-provided scratch
 *
 * Same Jacobi algorithm as the single-argument overload. On return:
 *   - A has eigenvalues on its diagonal (positions 0, 4, 8)
 *   - V holds eigenvectors as columns (row-major)
 *
 * @param Ain Input symmetric matrix (9 elements, row-major)
 * @param A   Scratch buffer (9 doubles), overwritten with eigenvalues on diagonal
 * @param V   Scratch buffer (9 doubles), overwritten with eigenvectors as columns
 */
inline void eigen_symmetric_3x3(const double* Ain, double* A, double* V) {
  std::memcpy(A, Ain, 9 * sizeof(double));
  V[0] = 1; V[1] = 0; V[2] = 0;
  V[3] = 0; V[4] = 1; V[5] = 0;
  V[6] = 0; V[7] = 0; V[8] = 1;

  for (int iter = 0; iter < 24; ++iter) {
    int p = 0, q = 1;
    double max_abs = std::abs(A[1]);
    if (std::abs(A[2]) > max_abs) { p = 0; q = 2; max_abs = std::abs(A[2]); }
    if (std::abs(A[5]) > max_abs) { p = 1; q = 2; max_abs = std::abs(A[5]); }
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
      A[kp] = c * akp - s * akq;  A[pk] = A[kp];
      A[kq] = s * akp + c * akq;  A[qk] = A[kq];
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
}
```

- [ ] **Step 4: Verify the math header compiles**

```bash
cd /home/merlot/codes/SplatLib && cmake --build build --target splat 2>&1 | tail -10
```

Expected: clean compile.

- [ ] **Step 5: Commit**

```bash
git add include/splat/maths/gaussian-decimate-math.h
git commit -m "feat(decimate): add ellipsoid_area() and caller-buffer eigen_symmetric_3x3 overload

Preparation for area-weighted merge math (#241 follow-up).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Switch SplatCache to Float32, remove Rt, use area-weighted mass

**Files:**
- Modify: `src/op/decimate.cpp:161-222`

- [ ] **Step 1: Change SplatCache struct to float and remove Rt**

Replace `struct SplatCache` (lines 161-169):

```cpp
struct SplatCache {
  std::vector<float> R;       // row-major 3x3 rotation per splat (f32)
  std::vector<float> v;       // variances (scale^2 + eps) per axis (f32)
  std::vector<float> invdiag; // 1/v per axis, precomputed (f32)
  std::vector<float> logdet;  // log-determinant per splat (f32)
  std::vector<float> sigma;   // full 9-element covariance per splat (f32)
  std::vector<float> mass;    // area·α weight per splat (f32)
};
```

- [ ] **Step 2: Rewrite `build_per_splat_cache()`**

Replace lines 171-222:

```cpp
SplatCache build_per_splat_cache(size_t n, const std::vector<float>& cx, const std::vector<float>& cy,
                                 const std::vector<float>& cz, const std::vector<float>& cop,
                                 const std::vector<float>& cs0, const std::vector<float>& cs1,
                                 const std::vector<float>& cs2, const std::vector<float>& cr0,
                                 const std::vector<float>& cr1, const std::vector<float>& cr2,
                                 const std::vector<float>& cr3) {
  SplatCache c;
  c.R.resize(n * 9);
  c.v.resize(n * 3);
  c.invdiag.resize(n * 3);
  c.logdet.resize(n);
  c.sigma.resize(n * 9);
  c.mass.resize(n);

  for (size_t i = 0; i < n; ++i) {
    const size_t i3 = 3 * i;
    const size_t i9 = 9 * i;

    const float lin_alpha = splat::sigmoid(cop[i]);
    const float sx = std::max(std::exp(cs0[i]), 1e-12f);
    const float sy = std::max(std::exp(cs1[i]), 1e-12f);
    const float sz = std::max(std::exp(cs2[i]), 1e-12f);

    const float vx = sx * sx + static_cast<float>(dm::kEpsCov);
    const float vy = sy * sy + static_cast<float>(dm::kEpsCov);
    const float vz = sz * sz + static_cast<float>(dm::kEpsCov);

    c.v[i3] = vx;
    c.v[i3 + 1] = vy;
    c.v[i3 + 2] = vz;
    c.invdiag[i3] = 1.0f / std::max(vx, 1e-30f);
    c.invdiag[i3 + 1] = 1.0f / std::max(vy, 1e-30f);
    c.invdiag[i3 + 2] = 1.0f / std::max(vz, 1e-30f);
    c.logdet[i] = std::log(std::max(vx, 1e-30f)) + std::log(std::max(vy, 1e-30f))
                  + std::log(std::max(vz, 1e-30f));

    // Normalize quaternion, build rotation in double for precision
    double qw = static_cast<double>(cr0[i]), qx = static_cast<double>(cr1[i]);
    double qy = static_cast<double>(cr2[i]), qz = static_cast<double>(cr3[i]);
    const double qn = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    const double invq = 1 / std::max(qn, 1e-12);
    qw *= invq; qx *= invq; qy *= invq; qz *= invq;

    double Rtmp[9];
    dm::quat_to_rotmat(qw, qx, qy, qz, Rtmp, 0);
    for (int k = 0; k < 9; ++k) c.R[i9 + static_cast<size_t>(k)] = static_cast<float>(Rtmp[k]);

    double sigtmp[9];
    dm::sigma_from_rot_var(Rtmp, 0, static_cast<double>(vx), static_cast<double>(vy),
                           static_cast<double>(vz), sigtmp, 0);
    for (int k = 0; k < 9; ++k) c.sigma[i9 + static_cast<size_t>(k)] = static_cast<float>(sigtmp[k]);

    // Area·α weighting (replaces volume·α NanoGS formula)
    c.mass[i] = lin_alpha * static_cast<float>(dm::ellipsoid_area(
        static_cast<double>(sx), static_cast<double>(sy), static_cast<double>(sz))) + 1e-12f;
  }
  return c;
}
```

- [ ] **Step 3: Commit**

```bash
git add src/op/decimate.cpp
git commit -m "refactor(decimate): switch SplatCache to Float32, remove Rt, use area-weighted mass

Memory ~50% lower for cache arrays. Mass now uses ellipsoid surface area
instead of volume for better anisotropic handling (#241).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Add MergeScratch and rewrite `compute_edge_cost()`

**Files:**
- Modify: `src/op/decimate.cpp:224-325`

- [ ] **Step 1: Add MergeScratch struct**

Insert before `compute_edge_cost` (before line 224):

```cpp
// Reusable scratch buffers for compute_edge_cost and moment_match.
// Allocated once per simplifyGaussians invocation — avoids per-call
// heap allocations in the merge loop.
struct MergeScratch {
  double sigm[9];  // compute_edge_cost: merged covariance
  double sigI[9];  // moment_match: input i covariance
  double sigJ[9];  // moment_match: input j covariance
  double rI[9];    // moment_match: input i rotation
  double rJ[9];    // moment_match: input j rotation
  double sig[9];   // moment_match: merged covariance
  double rM[9];    // moment_match: merged rotation
  double eigA[9];  // eigen_symmetric_3x3: working matrix
  double eigV[9];  // eigen_symmetric_3x3: eigenvectors
};
```

- [ ] **Step 2: Rewrite `compute_edge_cost()` signature and body**

Replace lines 224-325:

```cpp
double compute_edge_cost(size_t i, size_t j, const std::vector<float>& cx, const std::vector<float>& cy,
                         const std::vector<float>& cz, const SplatCache& cache,
                         const std::vector<std::array<double, 3>>& Z,
                         const std::vector<const std::vector<float>*>& app_cols,
                         MergeScratch& scratch) {
  const size_t i3 = 3 * i, j3 = 3 * j;
  const size_t i9 = 9 * i, j9 = 9 * j;

  const double mux = cx[i], muy = cy[i], muz = cz[i];
  const double mvx = cx[j], mvy = cy[j], mvz = cz[j];

  const double wi = cache.mass[i], wj = cache.mass[j];
  const double W = wi + wj;
  const double pi = wi / (W > 0 ? W : 1.0);
  const double pj = wj / (W > 0 ? W : 1.0);
  const double log_pi = std::log(std::max(pi, 1e-12));
  const double log_pj = std::log(std::max(pj, 1e-12));

  const double mmx = pi * mux + pj * mvx;
  const double mmy = pi * muy + pj * mvy;
  const double mmz = pi * muz + pj * mvz;

  const double dix = mux - mmx, diy = muy - mmy, diz = muz - mmz;
  const double djx = mvx - mmx, djy = mvy - mmy, djz = mvz - mmz;

  // Merged covariance (reuse scratch buffer)
  double* sigm = scratch.sigm;
  for (int a = 0; a < 9; ++a) {
    sigm[a] = pi * cache.sigma[i9 + a] + pj * cache.sigma[j9 + a];
  }
  sigm[0] += pi * dix * dix + pj * djx * djx;
  sigm[1] += pi * dix * diy + pj * djx * djy;
  sigm[2] += pi * dix * diz + pj * djx * djz;
  sigm[3] += pi * diy * dix + pj * djy * djx;
  sigm[4] += pi * diy * diy + pj * djy * djy;
  sigm[5] += pi * diy * diz + pj * djy * djz;
  sigm[6] += pi * diz * dix + pj * djz * djx;
  sigm[7] += pi * diz * diy + pj * djz * djy;
  sigm[8] += pi * diz * diz + pj * djz * djz;

  sigm[1] = sigm[3] = 0.5 * (sigm[1] + sigm[3]);
  sigm[2] = sigm[6] = 0.5 * (sigm[2] + sigm[6]);
  sigm[5] = sigm[7] = 0.5 * (sigm[5] + sigm[7]);
  sigm[0] += dm::kEpsCov; sigm[4] += dm::kEpsCov; sigm[8] += dm::kEpsCov;

  const double detm = std::max(dm::det3(sigm, 0), 1e-30);
  const double logdetm = std::log(detm);
  const double ep_neg_log_q = 0.5 * (3 * dm::kLog2Pi + logdetm + 3);

  const double stdix = std::sqrt(std::max(static_cast<double>(cache.v[i3]), 0.0));
  const double stdiy = std::sqrt(std::max(static_cast<double>(cache.v[i3 + 1]), 0.0));
  const double stdiz = std::sqrt(std::max(static_cast<double>(cache.v[i3 + 2]), 0.0));
  const double stdjx = std::sqrt(std::max(static_cast<double>(cache.v[j3]), 0.0));
  const double stdjy = std::sqrt(std::max(static_cast<double>(cache.v[j3 + 1]), 0.0));
  const double stdjz = std::sqrt(std::max(static_cast<double>(cache.v[j3 + 2]), 0.0));

  // Convert R to double once per edge (not per sample)
  double Ri_d[9], Rj_d[9];
  for (int k = 0; k < 9; ++k) {
    Ri_d[k] = cache.R[i9 + k];
    Rj_d[k] = cache.R[j9 + k];
  }

  // Sample x = mu + R · diag(std) · z  (uses R directly, no Rt)
  double sum_logp_on_i = 0, sum_logp_on_j = 0;
  for (const auto& zs : Z) {
    const double z0 = zs[0], z1 = zs[1], z2 = zs[2];

    const double xix = mux + z0 * stdix * Ri_d[0] + z1 * stdiy * Ri_d[1] + z2 * stdiz * Ri_d[2];
    const double xiy = muy + z0 * stdix * Ri_d[3] + z1 * stdiy * Ri_d[4] + z2 * stdiz * Ri_d[5];
    const double xiz = muz + z0 * stdix * Ri_d[6] + z1 * stdiy * Ri_d[7] + z2 * stdiz * Ri_d[8];

    const double xjx = mvx + z0 * stdjx * Rj_d[0] + z1 * stdjy * Rj_d[1] + z2 * stdjz * Rj_d[2];
    const double xjy = mvy + z0 * stdjx * Rj_d[3] + z1 * stdjy * Rj_d[4] + z2 * stdjz * Rj_d[5];
    const double xjz = mvz + z0 * stdjx * Rj_d[6] + z1 * stdjy * Rj_d[7] + z2 * stdjz * Rj_d[8];

    const double log_ni_on_i = dm::gauss_logpdf_diagrot(
        xix, xiy, xiz, mux, muy, muz, Ri_d, 0,
        cache.invdiag[i3], cache.invdiag[i3 + 1], cache.invdiag[i3 + 2], cache.logdet[i]);
    const double log_nj_on_i = dm::gauss_logpdf_diagrot(
        xix, xiy, xiz, mvx, mvy, mvz, Rj_d, 0,
        cache.invdiag[j3], cache.invdiag[j3 + 1], cache.invdiag[j3 + 2], cache.logdet[j]);
    sum_logp_on_i += dm::log_add_exp(log_pi + log_ni_on_i, log_pj + log_nj_on_i);

    const double log_ni_on_j = dm::gauss_logpdf_diagrot(
        xjx, xjy, xjz, mux, muy, muz, Ri_d, 0,
        cache.invdiag[i3], cache.invdiag[i3 + 1], cache.invdiag[i3 + 2], cache.logdet[i]);
    const double log_nj_on_j = dm::gauss_logpdf_diagrot(
        xjx, xjy, xjz, mvx, mvy, mvz, Rj_d, 0,
        cache.invdiag[j3], cache.invdiag[j3 + 1], cache.invdiag[j3 + 2], cache.logdet[j]);
    sum_logp_on_j += dm::log_add_exp(log_pi + log_ni_on_j, log_pj + log_nj_on_j);
  }

  const double n_s = static_cast<double>(Z.size());
  const double ei = sum_logp_on_i / n_s;
  const double ej = sum_logp_on_j / n_s;
  const double ep_logp = pi * ei + pj * ej;
  const double geo = ep_logp + ep_neg_log_q;

  double c_sh = 0;
  for (const std::vector<float>* col : app_cols) {
    const double d = static_cast<double>((*col)[i]) - static_cast<double>((*col)[j]);
    c_sh += d * d;
  }
  return geo + c_sh;
}
```

Key changes from old:
- Takes `MergeScratch& scratch` — sigm buffer comes from scratch
- `cache.R` is `float`, converted to `double Ri_d[9]`/`Rj_d[9]` once per edge
- Uses `R` directly (`x = mu + R·diag(std)·z`) instead of `Rt`
- `pi`/`pj` no longer clamped to [1e-12, 1-1e-12] (upstream dropped that)

- [ ] **Step 3: Commit**

```bash
git add src/op/decimate.cpp
git commit -m "refactor(decimate): add MergeScratch, rewrite compute_edge_cost to use R directly

Replaces per-call array allocation with scratch buffer. Switches from
Rt-based sampling to R-based sampling (matching upstream GPU kernel).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Rewrite `moment_match()` with area-weighted, mass-conserving merge math

**Files:**
- Modify: `src/op/decimate.cpp:327-430`

- [ ] **Step 1: Replace `moment_match()` entirely**

Replace lines 327-430:

```cpp
// Pairwise merge derived from sparkjsdev/spark gsplat.rs::new_merged:
//   - weights are α · ellipsoid-area
//   - merged covariance is area·α-weighted sum of (δδᵀ + Σ_k)
//   - merged opacity is mass-conserving, clamped at 1
//   - color is area·α weighted normalized average
void moment_match(size_t i, size_t j, const std::vector<float>& cx, const std::vector<float>& cy,
                  const std::vector<float>& cz, const std::vector<float>& cop, const std::vector<float>& cs0,
                  const std::vector<float>& cs1, const std::vector<float>& cs2, const std::vector<float>& cr0,
                  const std::vector<float>& cr1, const std::vector<float>& cr2, const std::vector<float>& cr3,
                  std::array<double, 3>& mu, std::array<double, 3>& sc, std::array<double, 4>& q, double& op_lin,
                  std::vector<double>& sh_out, const std::vector<const std::vector<float>*>& app_cols,
                  MergeScratch& scratch) {
  const double sxi = std::max(static_cast<double>(std::exp(cs0[i])), 1e-12);
  const double syi = std::max(static_cast<double>(std::exp(cs1[i])), 1e-12);
  const double szi = std::max(static_cast<double>(std::exp(cs2[i])), 1e-12);
  const double sxj = std::max(static_cast<double>(std::exp(cs0[j])), 1e-12);
  const double syj = std::max(static_cast<double>(std::exp(cs1[j])), 1e-12);
  const double szj = std::max(static_cast<double>(std::exp(cs2[j])), 1e-12);

  const double alpha_i = splat::sigmoid(cop[i]);
  const double alpha_j = splat::sigmoid(cop[j]);

  // Area·α weighting (replaces old volume·α NanoGS formula)
  const double Ai = dm::ellipsoid_area(sxi, syi, szi);
  const double Aj = dm::ellipsoid_area(sxj, syj, szj);
  const double wi = alpha_i * Ai + 1e-30;
  const double wj = alpha_j * Aj + 1e-30;
  const double W = wi + wj;
  const double pi = wi / W;
  const double pj = wj / W;

  // Merged mean (area·α weighted)
  const double mxi = cx[i], myi = cy[i], mzi = cz[i];
  const double mxj = cx[j], myj = cy[j], mzj = cz[j];
  const double mux = pi * mxi + pj * mxj;
  const double muy = pi * myi + pj * myj;
  const double muz = pi * mzi + pj * mzj;

  // Per-call scratch buffers
  double* SigI = scratch.sigI;
  double* SigJ = scratch.sigJ;
  double* Ri = scratch.rI;
  double* Rj = scratch.rJ;

  double qwi = cr0[i], qxi = cr1[i], qyi = cr2[i], qzi = cr3[i];
  double ni = 1 / std::max(std::sqrt(qwi * qwi + qxi * qxi + qyi * qyi + qzi * qzi), 1e-12);
  qwi *= ni; qxi *= ni; qyi *= ni; qzi *= ni;

  double qwj = cr0[j], qxj = cr1[j], qyj = cr2[j], qzj = cr3[j];
  double nj = 1 / std::max(std::sqrt(qwj * qwj + qxj * qxj + qyj * qyj + qzj * qzj), 1e-12);
  qwj *= nj; qxj *= nj; qyj *= nj; qzj *= nj;

  dm::quat_to_rotmat(qwi, qxi, qyi, qzi, Ri, 0);
  dm::quat_to_rotmat(qwj, qxj, qyj, qzj, Rj, 0);
  dm::sigma_from_rot_var(Ri, 0, sxi * sxi, syi * syi, szi * szi, SigI, 0);
  dm::sigma_from_rot_var(Rj, 0, sxj * sxj, syj * syj, szj * szj, SigJ, 0);

  const double dix = mxi - mux, diy = myi - muy, diz = mzi - muz;
  const double djx = mxj - mux, djy = myj - muy, djz = mzj - muz;

  // Merged covariance: weighted sum of (δδᵀ + Σ_k) — scratch
  double* Sig = scratch.sig;
  Sig[0] = pi * (dix * dix + SigI[0]) + pj * (djx * djx + SigJ[0]);
  Sig[1] = pi * (dix * diy + SigI[1]) + pj * (djx * djy + SigJ[1]);
  Sig[2] = pi * (dix * diz + SigI[2]) + pj * (djx * djz + SigJ[2]);
  Sig[3] = Sig[1];
  Sig[4] = pi * (diy * diy + SigI[4]) + pj * (djy * djy + SigJ[4]);
  Sig[5] = pi * (diy * diz + SigI[5]) + pj * (djy * djz + SigJ[5]);
  Sig[6] = Sig[2];
  Sig[7] = Sig[5];
  Sig[8] = pi * (diz * diz + SigI[8]) + pj * (djz * djz + SigJ[8]);
  Sig[0] += dm::kEpsCov; Sig[4] += dm::kEpsCov; Sig[8] += dm::kEpsCov;

  // Eigendecompose → scales (= √λ) and rotation
  double* eigA = scratch.eigA;
  double* eigV = scratch.eigV;
  dm::eigen_symmetric_3x3(Sig, eigA, eigV);

  // Sort eigenvalue indices descending (hand-unrolled, no std::sort alloc)
  const double v0 = eigA[0], v1 = eigA[4], v2 = eigA[8];
  int o0, o1, o2;
  if (v0 >= v1) {
    if (v1 >= v2)       { o0 = 0; o1 = 1; o2 = 2; }
    else if (v0 >= v2)  { o0 = 0; o1 = 2; o2 = 1; }
    else                { o0 = 2; o1 = 0; o2 = 1; }
  } else {
    if (v0 >= v2)       { o0 = 1; o1 = 0; o2 = 2; }
    else if (v1 >= v2)  { o0 = 1; o1 = 2; o2 = 0; }
    else                { o0 = 2; o1 = 1; o2 = 0; }
  }
  const double ev0 = std::max(eigA[3 * o0 + o0], 1e-18);
  const double ev1 = std::max(eigA[3 * o1 + o1], 1e-18);
  const double ev2 = std::max(eigA[3 * o2 + o2], 1e-18);
  const double s0 = std::sqrt(ev0);
  const double s1 = std::sqrt(ev1);
  const double s2 = std::sqrt(ev2);

  // Mass-conserving opacity, capped at 1
  const double alpha_m = std::min(1.0, W / std::max(dm::ellipsoid_area(s0, s1, s2), 1e-30));

  // Build rotation matrix from sorted eigenvectors (right-handed) — scratch
  double* Rm = scratch.rM;
  Rm[0] = eigV[o0]; Rm[1] = eigV[o1]; Rm[2] = eigV[o2];
  Rm[3] = eigV[3 + o0]; Rm[4] = eigV[3 + o1]; Rm[5] = eigV[3 + o2];
  Rm[6] = eigV[6 + o0]; Rm[7] = eigV[6 + o1]; Rm[8] = eigV[6 + o2];
  if (dm::det3(Rm, 0) < 0) {
    Rm[2] *= -1; Rm[5] *= -1; Rm[8] *= -1;
  }

  double qtmp[4];
  dm::rotmat_to_quat(Rm, 0, qtmp);

  mu = {mux, muy, muz};
  sc = {std::log(s0), std::log(s1), std::log(s2)};
  q = {qtmp[0], qtmp[1], qtmp[2], qtmp[3]};
  op_lin = alpha_m;

  // Color: area·α weighted normalized average
  sh_out.resize(app_cols.size());
  for (size_t k = 0; k < app_cols.size(); ++k) {
    sh_out[k] = pi * static_cast<double>((*app_cols[k])[i])
              + pj * static_cast<double>((*app_cols[k])[j]);
  }
}
```

Formula changes vs. old:

| Aspect | Old | New |
|--------|-----|-----|
| Weight | `(2π)^1.5·sx·sy·sz·α` | `ellipsoidArea(sx,sy,sz)·α` |
| Mean | mass-weighted `wi/W` | area·α-weighted `pi = wi/W` |
| Opacity | `α_i + α_j - α_i·α_j` | `min(1, W / area_merged)` |
| Color | `(wi·v_i + wj·v_j) / W` | `pi·v_i + pj·v_j` |
| Scale | `0.5·log(eigenvalue)` | `log(sqrt(eigenvalue))` |
| Eigen sort | `std::sort` with lambda | Hand-unrolled branch tree |
| Buffers | Per-call `std::array` | Pre-allocated `MergeScratch` |

- [ ] **Step 2: Commit**

```bash
git add src/op/decimate.cpp
git commit -m "fix(decimate): port area-weighted, mass-conserving merge math from upstream (#241)

Replaces NanoGS Porter-Duff opacity + volume-weighted formula with the
spark area-weighted variant. Fixes color drift and over-saturation at
aggressive reduction ratios.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Remove pre-merge opacity pruning, instantiate MergeScratch, update call sites

**Files:**
- Modify: `src/op/decimate.cpp` (body of `simplifyGaussians`)

- [ ] **Step 1: Remove unused constant**

Delete line 28:

```cpp
constexpr float kOpacityPruneThreshold = 0.1f;
```

- [ ] **Step 2: Remove opacity pruning block**

Replace lines 483-505 (the opacity pruning logic):

```cpp
  // No pre-merge opacity pruning — upstream dropped it (v2.2.0+).
  // The area·α merge math handles low-opacity splats through natural
  // weight attenuation.
  std::unique_ptr<DataTable> current = data_table.clone();
```

- [ ] **Step 3: Add MergeScratch instantiation**

After the Z samples line, add the scratch instance. Replace:

```cpp
  const std::vector<std::array<double, 3>> Z = dm::make_gaussian_samples(kMcSamples, 0);
  const bool profile = decimate_profile_enabled();
```

With:

```cpp
  const std::vector<std::array<double, 3>> Z = dm::make_gaussian_samples(kMcSamples, 0);

  // Per-call merge scratch — lives for the entire simplifyGaussians invocation.
  MergeScratch merge_scratch;

  const bool profile = decimate_profile_enabled();
```

- [ ] **Step 4: Update `compute_edge_cost` call site**

Replace:

```cpp
      costs[e] = static_cast<float>(compute_edge_cost(edge_u[e], edge_v[e], cx, cy, cz, cache, Z, app_data));
```

With:

```cpp
      costs[e] = static_cast<float>(compute_edge_cost(edge_u[e], edge_v[e], cx, cy, cz, cache, Z, app_data,
                                                       merge_scratch));
```

- [ ] **Step 5: Update `moment_match` call site**

Replace:

```cpp
      moment_match(pi, pj, cx, cy, cz, cop, cs0, cs1, cs2, cr0, cr1, cr2, cr3, mu, sc, q, op_lin, sh_merge,
                   app_data);
```

With:

```cpp
      moment_match(pi, pj, cx, cy, cz, cop, cs0, cs1, cs2, cr0, cr1, cr2, cr3, mu, sc, q, op_lin, sh_merge,
                   app_data, merge_scratch);
```

- [ ] **Step 6: Commit**

```bash
git add src/op/decimate.cpp
git commit -m "refactor(decimate): remove pre-merge opacity pruning, instantiate MergeScratch

Upstream dropped opacity pruning in v2.2.0. MergeScratch allocated once
per simplifyGaussians call and passed to compute_edge_cost/moment_match.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: Build, run tests, verify

**Files:**
- Test: `tests/decimate_test.cpp`

- [ ] **Step 1: Build the project**

```bash
cd /home/merlot/codes/SplatLib && cmake --build build --target splat transform 2>&1 | tail -20
```

Expected: clean build, no warnings.

- [ ] **Step 2: Run existing decimate test**

```bash
cd /home/merlot/codes/SplatLib && ./build/tests/decimate_test
```

Expected: exit code 0 (assertions pass: row count matches target, all columns finite).

- [ ] **Step 3: Run a smoke test with the transform CLI**

```bash
cd /home/merlot/codes/SplatLib && echo "Checking transform CLI" && ./build/transform/transform --help 2>&1 | head -5
```

- [ ] **Step 4: Commit any final adjustments**

```bash
git add tests/decimate_test.cpp
git commit -m "test(decimate): verify structural invariants hold after merge math change

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```
