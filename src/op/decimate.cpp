#include <splat/maths/gaussian-decimate-math.h>
#include <splat/maths/maths.h>
#include <splat/models/data-table.h>
#include <splat/op/decimate.h>
#include <splat/spatial/kdtree.h>
#include <splat/utils/logger.h>

#if defined(SPLAT_HAS_NANOFLANN)
#include <nanoflann.hpp>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace splat {
namespace {

namespace dm = decimate_math;

constexpr int kKnnK = 16;
constexpr int kMcSamples = 1;

constexpr const char* kRequiredCols[] = {"x",         "y",         "z",         "opacity",   "scale_0",
                                         "scale_1",   "scale_2",   "rot_0",     "rot_1",     "rot_2",
                                         "rot_3"};

using Clock = std::chrono::steady_clock;

enum class KnnBackend {
  BuiltIn,
  Nanoflann,
};

bool decimate_profile_enabled() {
  const char* value = std::getenv("SPLAT_DECIMATE_PROFILE");
  if (!value || value[0] == '\0') {
    return false;
  }
  const std::string text(value);
  return text != "0" && text != "false" && text != "FALSE";
}

KnnBackend decimate_knn_backend() {
#if defined(SPLAT_HAS_NANOFLANN)
  const char* value = std::getenv("SPLAT_DECIMATE_KNN");
  if (value && value[0] != '\0') {
    const std::string text(value);
    if (text == "legacy" || text == "builtin" || text == "built-in" || text == "kdtree") {
      return KnnBackend::BuiltIn;
    }
  }
  return KnnBackend::Nanoflann;
#else
  return KnnBackend::BuiltIn;
#endif
}

const char* knn_backend_name(KnnBackend backend) {
  switch (backend) {
    case KnnBackend::Nanoflann:
      return "nanoflann";
    case KnnBackend::BuiltIn:
    default:
      return "built-in";
  }
}

double elapsed_ms(Clock::time_point start, Clock::time_point end = Clock::now()) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

#if defined(SPLAT_HAS_NANOFLANN)
struct NanoflannPointCloud {
  const std::vector<float>& x;
  const std::vector<float>& y;
  const std::vector<float>& z;

  size_t kdtree_get_point_count() const {
    return x.size();
  }

  float kdtree_get_pt(size_t idx, size_t dim) const {
    if (dim == 0) {
      return x[idx];
    }
    if (dim == 1) {
      return y[idx];
    }
    return z[idx];
  }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const {
    return false;
  }
};

using NanoflannIndex =
    nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, NanoflannPointCloud>,
                                        NanoflannPointCloud, 3, size_t>;
#endif

bool has_required_columns(const DataTable& dt) {
  for (const char* name : kRequiredCols) {
    if (!dt.hasColumn(name)) {
      return false;
    }
  }
  return true;
}

void collect_appearance_columns(const DataTable& dt, std::vector<std::string>& out) {
  out.clear();
  for (const char* name : {"f_dc_0", "f_dc_1", "f_dc_2"}) {
    if (dt.hasColumn(name)) {
      out.push_back(name);
    }
  }
  for (int i = 0; i < 45; ++i) {
    const std::string name = "f_rest_" + std::to_string(i);
    if (dt.hasColumn(name)) {
      out.push_back(name);
    }
  }
}

void copy_cell(Column& dst, size_t di, const Column& src, size_t si) {
  std::visit(
      [&](auto& dvec) {
        using T = typename std::decay_t<decltype(dvec)>::value_type;
        const auto& svec = src.asVector<T>();
        dvec[di] = svec[si];
      },
      dst.data);
}

std::unique_ptr<DataTable> empty_table_like(const DataTable& src, size_t num_rows) {
  std::vector<Column> new_columns;
  new_columns.reserve(src.columns.size());
  for (const auto& c : src.columns) {
    TypedArray nd = std::visit(
        [num_rows](const auto& vec) -> TypedArray {
          using T = typename std::decay_t<decltype(vec)>::value_type;
          return std::vector<T>(num_rows);
        },
        c.data);
    new_columns.emplace_back(Column{c.name, std::move(nd)});
  }
  return std::make_unique<DataTable>(new_columns);
}

struct SplatCache {
  std::vector<float> R;       // row-major 3x3 rotation per splat (f32)
  std::vector<float> v;       // variances (scale^2 + eps) per axis (f32)
  std::vector<float> invdiag; // 1/v per axis, precomputed (f32)
  std::vector<float> logdet;  // log-determinant per splat (f32)
  std::vector<float> sigma;   // full 9-element covariance per splat (f32)
  std::vector<float> mass;    // area*alpha weight per splat (f32)
};

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

    // Area*alpha weighting (replaces volume*alpha NanoGS formula)
    c.mass[i] = lin_alpha * static_cast<float>(dm::ellipsoid_area(
        static_cast<double>(sx), static_cast<double>(sy), static_cast<double>(sz))) + 1e-12f;
  }
  return c;
}

// Reusable scratch buffers for compute_edge_cost and moment_match.
// Allocated once per simplifyGaussians invocation -- avoids per-call
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

  // Sample x = mu + R . diag(std) . z  (uses R directly, no Rt)
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

  // Area-alpha weighting (replaces old volume-alpha NanoGS formula)
  const double Ai = dm::ellipsoid_area(sxi, syi, szi);
  const double Aj = dm::ellipsoid_area(sxj, syj, szj);
  const double wi = alpha_i * Ai + 1e-30;
  const double wj = alpha_j * Aj + 1e-30;
  const double W = wi + wj;
  const double pi = wi / W;
  const double pj = wj / W;

  // Merged mean (area-alpha weighted)
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

  // Merged covariance: weighted sum of (delta*delta' + Sigma_k) -- scratch
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

  // Eigendecompose -> scales (= sqrt(lambda)) and rotation
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

  // Build rotation matrix from sorted eigenvectors (right-handed) -- scratch
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

  // Color: area-alpha weighted normalized average
  sh_out.resize(app_cols.size());
  for (size_t k = 0; k < app_cols.size(); ++k) {
    sh_out[k] = pi * static_cast<double>((*app_cols[k])[i])
              + pj * static_cast<double>((*app_cols[k])[j]);
  }
}

}  // namespace

void sortByVisibility(const DataTable& data_table, std::vector<uint32_t>& indices) {
  if (indices.empty()) {
    return;
  }
  if (!data_table.hasColumn("opacity") || !data_table.hasColumn("scale_0") || !data_table.hasColumn("scale_1") ||
      !data_table.hasColumn("scale_2")) {
    return;
  }
  const auto& opacity = data_table.getColumnByName("opacity").asVector<float>();
  const auto& scale0 = data_table.getColumnByName("scale_0").asVector<float>();
  const auto& scale1 = data_table.getColumnByName("scale_1").asVector<float>();
  const auto& scale2 = data_table.getColumnByName("scale_2").asVector<float>();

  std::vector<size_t> order(indices.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    const size_t ra = indices[a], rb = indices[b];
    const float sa =
        splat::sigmoid(opacity[ra]) * std::exp(scale0[ra] + scale1[ra] + scale2[ra]);
    const float sb =
        splat::sigmoid(opacity[rb]) * std::exp(scale0[rb] + scale1[rb] + scale2[rb]);
    return sa > sb;
  });
  std::vector<uint32_t> tmp = indices;
  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = tmp[order[i]];
  }
}

std::unique_ptr<DataTable> simplifyGaussians(const DataTable& data_table, int targetCount) {
  const size_t N = data_table.getNumRows();
  LOG_INFO("simplifyGaussians start rows=%zu target=%d", N, targetCount);

  if (targetCount <= 0) {
    LOG_INFO("simplifyGaussians targetCount=%d, returning empty table", targetCount);
    return empty_table_like(data_table, 0);
  }
  if (N <= static_cast<size_t>(targetCount)) {
    LOG_INFO("simplifyGaussians source already within target rows=%zu", N);
    return data_table.clone();
  }

  if (!has_required_columns(data_table)) {
    LOG_WARN("simplifyGaussians missing required Gaussian columns, falling back to visibility sort");
    std::vector<uint32_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0u);
    sortByVisibility(data_table, indices);
    indices.resize(static_cast<size_t>(targetCount));
    LOG_INFO("simplifyGaussians fallback kept=%d dropped=%zu", targetCount, N - static_cast<size_t>(targetCount));
    return data_table.permuteRows(indices);
  }

  std::vector<std::string> appearance_names;
  collect_appearance_columns(data_table, appearance_names);

  std::unique_ptr<DataTable> current = data_table.clone();

  const std::vector<std::array<double, 3>> Z = dm::make_gaussian_samples(kMcSamples, 0);
  const bool profile = decimate_profile_enabled();
  size_t iteration = 0;

  while (current->getNumRows() > static_cast<size_t>(targetCount)) {
    ++iteration;
    const size_t n = current->getNumRows();
    const size_t k_eff = static_cast<size_t>(std::min(std::max(1, kKnnK), static_cast<int>(std::max<size_t>(1, n - 1))));
    const KnnBackend knn_backend = decimate_knn_backend();
    LOG_INFO("simplifyGaussians iteration=%zu rows=%zu target=%d k=%zu backend=%s", iteration, n, targetCount, k_eff,
             knn_backend_name(knn_backend));
    const auto iteration_start = Clock::now();
    auto phase_start = iteration_start;
    auto log_phase = [&](const char* name) {
      if (profile) {
        LOG_INFO("decimate[%zu] %-18s %.3f ms", iteration, name, elapsed_ms(phase_start));
      }
      phase_start = Clock::now();
    };
    auto log_total = [&]() {
      if (profile) {
        LOG_INFO("decimate[%zu] total rows=%zu target=%d elapsed=%.3f ms", iteration, n, targetCount,
                 elapsed_ms(iteration_start));
      }
    };

    if (profile) {
      LOG_INFO("decimate[%zu] start rows=%zu target=%d k=%zu backend=%s", iteration, n, targetCount, k_eff,
               knn_backend_name(knn_backend));
    }

    const auto& cx = current->getColumnByName("x").asVector<float>();
    const auto& cy = current->getColumnByName("y").asVector<float>();
    const auto& cz = current->getColumnByName("z").asVector<float>();
    const auto& cop = current->getColumnByName("opacity").asVector<float>();
    const auto& cs0 = current->getColumnByName("scale_0").asVector<float>();
    const auto& cs1 = current->getColumnByName("scale_1").asVector<float>();
    const auto& cs2 = current->getColumnByName("scale_2").asVector<float>();
    const auto& cr0 = current->getColumnByName("rot_0").asVector<float>();
    const auto& cr1 = current->getColumnByName("rot_1").asVector<float>();
    const auto& cr2 = current->getColumnByName("rot_2").asVector<float>();
    const auto& cr3 = current->getColumnByName("rot_3").asVector<float>();

    SplatCache cache = build_per_splat_cache(n, cx, cy, cz, cop, cs0, cs1, cs2, cr0, cr1, cr2, cr3);
    log_phase("cache");

    size_t edge_capacity = (n * k_eff) / 2 + 8;
    std::vector<uint32_t> edge_u(edge_capacity), edge_v(edge_capacity);
    size_t edge_count = 0;

    auto ensure_edge_cap = [&](size_t need) {
      if (need <= edge_capacity) {
        return;
      }
      while (edge_capacity < need) {
        edge_capacity *= 2;
      }
      edge_u.resize(edge_capacity);
      edge_v.resize(edge_capacity);
    };

#if defined(SPLAT_HAS_NANOFLANN)
    if (knn_backend == KnnBackend::Nanoflann) {
      NanoflannPointCloud points{cx, cy, cz};
      NanoflannIndex index(3, points, nanoflann::KDTreeSingleIndexAdaptorParams(16));
      log_phase("kd_build");

      float query_point[3] = {};
      std::vector<size_t> knn_indices(k_eff + 1);
      std::vector<float> knn_distances(k_eff + 1);
      for (size_t i = 0; i < n; ++i) {
        query_point[0] = cx[i];
        query_point[1] = cy[i];
        query_point[2] = cz[i];
        const size_t found = index.knnSearch(query_point, k_eff + 1, knn_indices.data(), knn_distances.data());
        for (size_t ki = 0; ki < found; ++ki) {
          const size_t kj = knn_indices[ki];
          if (kj <= i) {
            continue;
          }
          ensure_edge_cap(edge_count + 1);
          edge_u[edge_count] = static_cast<uint32_t>(i);
          edge_v[edge_count] = static_cast<uint32_t>(kj);
          ++edge_count;
        }
      }
    } else
#endif
    {
      std::vector<float> px(cx.begin(), cx.end());
      std::vector<float> py(cy.begin(), cy.end());
      std::vector<float> pz(cz.begin(), cz.end());
      std::vector<Column> pos_cols;
      pos_cols.push_back(Column{"x", std::move(px)});
      pos_cols.push_back(Column{"y", std::move(py)});
      pos_cols.push_back(Column{"z", std::move(pz)});
      DataTable pos_table(pos_cols);
      KdTree kd(&pos_table);
      log_phase("kd_build");

      float query_point[3] = {};
      std::vector<size_t> knn;
      knn.reserve(k_eff + 1);
      for (size_t i = 0; i < n; ++i) {
        query_point[0] = cx[i];
        query_point[1] = cy[i];
        query_point[2] = cz[i];
        kd.findKNearest(query_point, k_eff + 1, knn);
        for (size_t kj : knn) {
          if (kj <= i) {
            continue;
          }
          ensure_edge_cap(edge_count + 1);
          edge_u[edge_count] = static_cast<uint32_t>(i);
          edge_v[edge_count] = static_cast<uint32_t>(kj);
          ++edge_count;
        }
      }
    }
    log_phase("knn");
    if (profile) {
      LOG_INFO("decimate[%zu] candidate_edges=%zu", iteration, edge_count);
    }

    if (edge_count == 0) {
      LOG_WARN("simplifyGaussians iteration=%zu found no candidate edges, stopping at rows=%zu target=%d", iteration, n,
               targetCount);
      log_total();
      break;
    }

    std::vector<const std::vector<float>*> app_data;
    app_data.reserve(appearance_names.size());
    for (const std::string& name : appearance_names) {
      app_data.push_back(&current->getColumnByName(name).asVector<float>());
    }

    MergeScratch scratch;
    const size_t merges_needed = n - static_cast<size_t>(targetCount);
    std::vector<float> costs(edge_count);
    for (size_t e = 0; e < edge_count; ++e) {
      costs[e] = static_cast<float>(compute_edge_cost(edge_u[e], edge_v[e], cx, cy, cz, cache, Z, app_data, scratch));
    }
    log_phase("edge_cost");

    std::vector<uint32_t> sorted(edge_count);
    const size_t valid_count = dm::radix_sort_indices_by_float(sorted.data(), edge_count, costs.data());
    log_phase("sort");

    std::vector<uint8_t> used(n, 0);
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    pairs.reserve(merges_needed);
    for (size_t t = 0; t < valid_count && pairs.size() < merges_needed; ++t) {
      const uint32_t e = sorted[t];
      const uint32_t u = edge_u[e], v = edge_v[e];
      if (used[u] || used[v]) {
        continue;
      }
      used[u] = 1;
      used[v] = 1;
      pairs.emplace_back(u, v);
    }
    log_phase("pair_select");
    if (profile) {
      LOG_INFO("decimate[%zu] pairs=%zu valid_edges=%zu", iteration, pairs.size(), valid_count);
    }

    if (pairs.empty()) {
      LOG_WARN("simplifyGaussians iteration=%zu found no mergeable pairs among %zu valid edges, stopping at rows=%zu "
               "target=%d",
               iteration, valid_count, n, targetCount);
      log_total();
      break;
    }

    std::vector<uint8_t> used_set(n, 0);
    for (const auto& pr : pairs) {
      used_set[pr.first] = 1;
      used_set[pr.second] = 1;
    }

    std::vector<uint32_t> keep_indices;
    keep_indices.reserve(n - pairs.size());
    for (size_t i = 0; i < n; ++i) {
      if (!used_set[i]) {
        keep_indices.push_back(static_cast<uint32_t>(i));
      }
    }
    log_phase("keep_indices");

    const size_t out_count = keep_indices.size() + pairs.size();
    std::unique_ptr<DataTable> new_table = empty_table_like(*current, out_count);
    log_phase("table_alloc");

    size_t dst = 0;
    for (uint32_t src : keep_indices) {
      for (size_t c = 0; c < current->columns.size(); ++c) {
        copy_cell(new_table->columns[c], dst, current->columns[c], src);
      }
      ++dst;
    }
    log_phase("copy_keep");

    std::array<double, 3> mu{}, sc{};
    std::array<double, 4> q{};
    double op_lin = 0;
    std::vector<double> sh_merge;

    Column& dst_x_col = new_table->getColumnByName("x");
    Column& dst_y_col = new_table->getColumnByName("y");
    Column& dst_z_col = new_table->getColumnByName("z");
    Column& dst_s0 = new_table->getColumnByName("scale_0");
    Column& dst_s1 = new_table->getColumnByName("scale_1");
    Column& dst_s2 = new_table->getColumnByName("scale_2");
    Column& dst_r0 = new_table->getColumnByName("rot_0");
    Column& dst_r1 = new_table->getColumnByName("rot_1");
    Column& dst_r2 = new_table->getColumnByName("rot_2");
    Column& dst_r3 = new_table->getColumnByName("rot_3");
    Column& dst_op = new_table->getColumnByName("opacity");

    std::vector<Column*> dst_app;
    for (const std::string& name : appearance_names) {
      dst_app.push_back(&new_table->getColumnByName(name));
    }

    std::vector<bool> handled(current->columns.size(), false);
    auto mark_handled = [&](const std::string& name) {
      for (size_t ci = 0; ci < current->columns.size(); ++ci) {
        if (current->columns[ci].name == name) {
          handled[ci] = true;
          break;
        }
      }
    };
    for (const char* name : {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2",
                              "rot_3"}) {
      mark_handled(name);
    }
    for (const std::string& name : appearance_names) {
      mark_handled(name);
    }

    std::vector<std::pair<size_t, Column*>> unhandled;
    for (size_t ci = 0; ci < current->columns.size(); ++ci) {
      if (!handled[ci] && new_table->hasColumn(current->columns[ci].name)) {
        unhandled.emplace_back(ci, &new_table->getColumnByName(current->columns[ci].name));
      }
    }

    for (const auto& pr : pairs) {
      const uint32_t pi = pr.first, pj = pr.second;
      moment_match(pi, pj, cx, cy, cz, cop, cs0, cs1, cs2, cr0, cr1, cr2, cr3, mu, sc, q, op_lin, sh_merge,
                   app_data, scratch);

      dst_x_col.setValue(dst, static_cast<float>(mu[0]));
      dst_y_col.setValue(dst, static_cast<float>(mu[1]));
      dst_z_col.setValue(dst, static_cast<float>(mu[2]));
      dst_s0.setValue(dst, static_cast<float>(sc[0]));
      dst_s1.setValue(dst, static_cast<float>(sc[1]));
      dst_s2.setValue(dst, static_cast<float>(sc[2]));
      dst_r0.setValue(dst, static_cast<float>(q[0]));
      dst_r1.setValue(dst, static_cast<float>(q[1]));
      dst_r2.setValue(dst, static_cast<float>(q[2]));
      dst_r3.setValue(dst, static_cast<float>(q[3]));
      const double op_clamped = std::max(0.0, std::min(1.0, op_lin));
      dst_op.setValue(dst, static_cast<float>(dm::logit(op_clamped)));

      for (size_t k = 0; k < dst_app.size(); ++k) {
        dst_app[k]->setValue(dst, static_cast<float>(sh_merge[k]));
      }

      const size_t dominant = cache.mass[pi] >= cache.mass[pj] ? static_cast<size_t>(pi) : static_cast<size_t>(pj);
      for (const auto& uh : unhandled) {
        copy_cell(*uh.second, dst, current->columns[uh.first], dominant);
      }
      ++dst;
    }
    log_phase("merge_pairs");
    log_total();

    LOG_INFO("simplifyGaussians iteration=%zu complete rows=%zu -> %zu merges=%zu", iteration, n, new_table->getNumRows(),
             pairs.size());
    current = std::move(new_table);
  }

  LOG_INFO("simplifyGaussians done source=%zu target=%d iterations=%zu final=%zu", N, targetCount, iteration,
           current->getNumRows());
  return current;
}

}  // namespace splat
