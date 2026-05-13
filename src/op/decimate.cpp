#include <splat/maths/gaussian-decimate-math.h>
#include <splat/maths/maths.h>
#include <splat/models/data-table.h>
#include <splat/op/decimate.h>
#include <splat/spatial/kdtree.h>
#include <splat/utils/logger.h>

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

constexpr float kOpacityPruneThreshold = 0.1f;
constexpr int kKnnK = 16;
constexpr int kMcSamples = 1;

constexpr const char* kRequiredCols[] = {"x",         "y",         "z",         "opacity",   "scale_0",
                                         "scale_1",   "scale_2",   "rot_0",     "rot_1",     "rot_2",
                                         "rot_3"};

using Clock = std::chrono::steady_clock;

bool decimate_profile_enabled() {
  const char* value = std::getenv("SPLAT_DECIMATE_PROFILE");
  if (!value || value[0] == '\0') {
    return false;
  }
  const std::string text(value);
  return text != "0" && text != "false" && text != "FALSE";
}

double elapsed_ms(Clock::time_point start, Clock::time_point end = Clock::now()) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

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
  std::vector<double> R;
  std::vector<double> Rt;
  std::vector<double> v;
  std::vector<double> invdiag;
  std::vector<double> logdet;
  std::vector<double> sigma;
  std::vector<double> mass;
};

SplatCache build_per_splat_cache(size_t n, const std::vector<float>& cx, const std::vector<float>& cy,
                                 const std::vector<float>& cz, const std::vector<float>& cop,
                                 const std::vector<float>& cs0, const std::vector<float>& cs1,
                                 const std::vector<float>& cs2, const std::vector<float>& cr0,
                                 const std::vector<float>& cr1, const std::vector<float>& cr2,
                                 const std::vector<float>& cr3) {
  SplatCache c;
  c.R.resize(n * 9);
  c.Rt.resize(n * 9);
  c.v.resize(n * 3);
  c.invdiag.resize(n * 3);
  c.logdet.resize(n);
  c.sigma.resize(n * 9);
  c.mass.resize(n);

  for (size_t i = 0; i < n; ++i) {
    const size_t i3 = 3 * i;
    const size_t i9 = 9 * i;

    const double lin_alpha = static_cast<double>(splat::sigmoid(cop[i]));
    const double sx = std::max(static_cast<double>(std::exp(cs0[i])), 1e-12);
    const double sy = std::max(static_cast<double>(std::exp(cs1[i])), 1e-12);
    const double sz = std::max(static_cast<double>(std::exp(cs2[i])), 1e-12);

    const double vx = sx * sx + dm::kEpsCov;
    const double vy = sy * sy + dm::kEpsCov;
    const double vz = sz * sz + dm::kEpsCov;

    c.v[i3] = vx;
    c.v[i3 + 1] = vy;
    c.v[i3 + 2] = vz;
    c.invdiag[i3] = 1 / std::max(vx, 1e-30);
    c.invdiag[i3 + 1] = 1 / std::max(vy, 1e-30);
    c.invdiag[i3 + 2] = 1 / std::max(vz, 1e-30);
    c.logdet[i] = std::log(std::max(vx, 1e-30)) + std::log(std::max(vy, 1e-30)) + std::log(std::max(vz, 1e-30));

    double qw = cr0[i], qx = cr1[i], qy = cr2[i], qz = cr3[i];
    const double qn = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    const double invq = 1 / std::max(qn, 1e-12);
    qw *= invq;
    qx *= invq;
    qy *= invq;
    qz *= invq;

    dm::quat_to_rotmat(qw, qx, qy, qz, c.R.data(), i9);
    dm::transpose3(c.R.data(), i9, c.Rt.data(), i9);
    dm::sigma_from_rot_var(c.R.data(), i9, vx, vy, vz, c.sigma.data(), i9);

    c.mass[i] = dm::kTwoPiPow1_5 * lin_alpha * sx * sy * sz + 1e-12;
  }
  return c;
}

double compute_edge_cost(size_t i, size_t j, const std::vector<float>& cx, const std::vector<float>& cy,
                         const std::vector<float>& cz, const SplatCache& cache,
                         const std::vector<std::array<double, 3>>& Z,
                         const std::vector<const std::vector<float>*>& app_cols) {
  const size_t i3 = 3 * i, j3 = 3 * j;
  const size_t i9 = 9 * i, j9 = 9 * j;

  const double mux = cx[i], muy = cy[i], muz = cz[i];
  const double mvx = cx[j], mvy = cy[j], mvz = cz[j];

  const double wi = cache.mass[i], wj = cache.mass[j];
  const double W = wi + wj;
  const double Wsafe = W > 0 ? W : 1.0;

  double pi = wi / Wsafe;
  pi = std::max(1e-12, std::min(1.0 - 1e-12, pi));
  const double pj = 1.0 - pi;
  const double log_pi = std::log(pi);
  const double log_pj = std::log(pj);

  const double mmx = pi * mux + pj * mvx;
  const double mmy = pi * muy + pj * mvy;
  const double mmz = pi * muz + pj * mvz;

  const double dix = mux - mmx, diy = muy - mmy, diz = muz - mmz;
  const double djx = mvx - mmx, djy = mvy - mmy, djz = mvz - mmz;

  std::array<double, 9> sigm{};
  for (int a = 0; a < 9; ++a) {
    sigm[static_cast<size_t>(a)] = pi * cache.sigma[i9 + static_cast<size_t>(a)] +
                                   pj * cache.sigma[j9 + static_cast<size_t>(a)];
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
  sigm[0] += dm::kEpsCov;
  sigm[4] += dm::kEpsCov;
  sigm[8] += dm::kEpsCov;

  const double detm = std::max(dm::det3(sigm.data(), 0), 1e-30);
  const double logdetm = std::log(detm);
  const double ep_neg_log_q = 0.5 * (3 * dm::kLog2Pi + logdetm + 3);

  const double stdix = std::sqrt(std::max(cache.v[i3], 0.0));
  const double stdiy = std::sqrt(std::max(cache.v[i3 + 1], 0.0));
  const double stdiz = std::sqrt(std::max(cache.v[i3 + 2], 0.0));
  const double stdjx = std::sqrt(std::max(cache.v[j3], 0.0));
  const double stdjy = std::sqrt(std::max(cache.v[j3 + 1], 0.0));
  const double stdjz = std::sqrt(std::max(cache.v[j3 + 2], 0.0));

  double sum_logp_on_i = 0, sum_logp_on_j = 0;
  for (const auto& zs : Z) {
    const double z0 = zs[0], z1 = zs[1], z2 = zs[2];

    const double xix = mux + z0 * stdix * cache.Rt[i9] + z1 * stdiy * cache.Rt[i9 + 3] + z2 * stdiz * cache.Rt[i9 + 6];
    const double xiy = muy + z0 * stdix * cache.Rt[i9 + 1] + z1 * stdiy * cache.Rt[i9 + 4] + z2 * stdiz * cache.Rt[i9 + 7];
    const double xiz = muz + z0 * stdix * cache.Rt[i9 + 2] + z1 * stdiy * cache.Rt[i9 + 5] + z2 * stdiz * cache.Rt[i9 + 8];

    const double xjx = mvx + z0 * stdjx * cache.Rt[j9] + z1 * stdjy * cache.Rt[j9 + 3] + z2 * stdjz * cache.Rt[j9 + 6];
    const double xjy = mvy + z0 * stdjx * cache.Rt[j9 + 1] + z1 * stdjy * cache.Rt[j9 + 4] + z2 * stdjz * cache.Rt[j9 + 7];
    const double xjz = mvz + z0 * stdjx * cache.Rt[j9 + 2] + z1 * stdjy * cache.Rt[j9 + 5] + z2 * stdjz * cache.Rt[j9 + 8];

    const double log_ni_on_i =
        dm::gauss_logpdf_diagrot(xix, xiy, xiz, mux, muy, muz, cache.R.data(), i9, cache.invdiag[i3],
                                 cache.invdiag[i3 + 1], cache.invdiag[i3 + 2], cache.logdet[i]);
    const double log_nj_on_i =
        dm::gauss_logpdf_diagrot(xix, xiy, xiz, mvx, mvy, mvz, cache.R.data(), j9, cache.invdiag[j3],
                                 cache.invdiag[j3 + 1], cache.invdiag[j3 + 2], cache.logdet[j]);
    sum_logp_on_i += dm::log_add_exp(log_pi + log_ni_on_i, log_pj + log_nj_on_i);

    const double log_ni_on_j =
        dm::gauss_logpdf_diagrot(xjx, xjy, xjz, mux, muy, muz, cache.R.data(), i9, cache.invdiag[i3],
                                 cache.invdiag[i3 + 1], cache.invdiag[i3 + 2], cache.logdet[i]);
    const double log_nj_on_j =
        dm::gauss_logpdf_diagrot(xjx, xjy, xjz, mvx, mvy, mvz, cache.R.data(), j9, cache.invdiag[j3],
                                 cache.invdiag[j3 + 1], cache.invdiag[j3 + 2], cache.logdet[j]);
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
                  std::vector<double>& sh_out, const std::vector<const std::vector<float>*>& app_cols) {
  const double sxi = std::max(static_cast<double>(std::exp(cs0[i])), 1e-12);
  const double syi = std::max(static_cast<double>(std::exp(cs1[i])), 1e-12);
  const double szi = std::max(static_cast<double>(std::exp(cs2[i])), 1e-12);
  const double sxj = std::max(static_cast<double>(std::exp(cs0[j])), 1e-12);
  const double syj = std::max(static_cast<double>(std::exp(cs1[j])), 1e-12);
  const double szj = std::max(static_cast<double>(std::exp(cs2[j])), 1e-12);

  const double alpha_i = static_cast<double>(splat::sigmoid(cop[i]));
  const double alpha_j = static_cast<double>(splat::sigmoid(cop[j]));

  const double wi = dm::kTwoPiPow1_5 * alpha_i * sxi * syi * szi + 1e-12;
  const double wj = dm::kTwoPiPow1_5 * alpha_j * sxj * syj * szj + 1e-12;
  const double W = std::max(wi + wj, 1e-12);

  const double mux = (wi * cx[i] + wj * cx[j]) / W;
  const double muy = (wi * cy[i] + wj * cy[j]) / W;
  const double muz = (wi * cz[i] + wj * cz[j]) / W;

  std::array<double, 9> sig_i{}, sig_j{}, ri{}, rj{};
  double qwi = cr0[i], qxi = cr1[i], qyi = cr2[i], qzi = cr3[i];
  double ni = std::sqrt(qwi * qwi + qxi * qxi + qyi * qyi + qzi * qzi);
  ni = 1 / std::max(ni, 1e-12);
  qwi *= ni;
  qxi *= ni;
  qyi *= ni;
  qzi *= ni;

  double qwj = cr0[j], qxj = cr1[j], qyj = cr2[j], qzj = cr3[j];
  double nj = std::sqrt(qwj * qwj + qxj * qxj + qyj * qyj + qzj * qzj);
  nj = 1 / std::max(nj, 1e-12);
  qwj *= nj;
  qxj *= nj;
  qyj *= nj;
  qzj *= nj;

  dm::quat_to_rotmat(qwi, qxi, qyi, qzi, ri.data(), 0);
  dm::quat_to_rotmat(qwj, qxj, qyj, qzj, rj.data(), 0);
  dm::sigma_from_rot_var(ri.data(), 0, sxi * sxi, syi * syi, szi * szi, sig_i.data(), 0);
  dm::sigma_from_rot_var(rj.data(), 0, sxj * sxj, syj * syj, szj * szj, sig_j.data(), 0);

  const double dix = cx[i] - mux, diy = cy[i] - muy, diz = cz[i] - muz;
  const double djx = cx[j] - mux, djy = cy[j] - muy, djz = cz[j] - muz;

  std::array<double, 9> sig{};
  for (int a = 0; a < 9; ++a) {
    sig[static_cast<size_t>(a)] = (wi * sig_i[static_cast<size_t>(a)] + wj * sig_j[static_cast<size_t>(a)]) / W;
  }
  sig[0] += (wi * dix * dix + wj * djx * djx) / W;
  sig[1] += (wi * dix * diy + wj * djx * djy) / W;
  sig[2] += (wi * dix * diz + wj * djx * djz) / W;
  sig[3] += (wi * diy * dix + wj * djy * djx) / W;
  sig[4] += (wi * diy * diy + wj * djy * djy) / W;
  sig[5] += (wi * diy * diz + wj * djy * djz) / W;
  sig[6] += (wi * diz * dix + wj * djz * djx) / W;
  sig[7] += (wi * diz * diy + wj * djz * djy) / W;
  sig[8] += (wi * diz * diz + wj * djz * djz) / W;

  sig[1] = sig[3] = 0.5 * (sig[1] + sig[3]);
  sig[2] = sig[6] = 0.5 * (sig[2] + sig[6]);
  sig[5] = sig[7] = 0.5 * (sig[5] + sig[7]);
  sig[0] += dm::kEpsCov;
  sig[4] += dm::kEpsCov;
  sig[8] += dm::kEpsCov;

  dm::Symmetric3x3Eigen ev = dm::eigen_symmetric_3x3(sig.data());
  double vals[3] = {ev.values[0], ev.values[1], ev.values[2]};
  int order[3] = {0, 1, 2};
  std::sort(order, order + 3, [&](int a, int b) { return vals[a] > vals[b]; });
  const double sorted_vals[3] = {std::max(vals[order[0]], 1e-18), std::max(vals[order[1]], 1e-18),
                                 std::max(vals[order[2]], 1e-18)};

  std::array<double, 9> rm{};
  for (int c = 0; c < 3; ++c) {
    const int src = order[c];
    rm[static_cast<size_t>(c)] = ev.vectors[src];
    rm[3 + static_cast<size_t>(c)] = ev.vectors[3 + src];
    rm[6 + static_cast<size_t>(c)] = ev.vectors[6 + src];
  }

  if (dm::det3(rm.data(), 0) < 0) {
    rm[2] *= -1;
    rm[5] *= -1;
    rm[8] *= -1;
  }

  std::array<double, 4> qtmp{};
  dm::rotmat_to_quat(rm.data(), 0, qtmp.data());

  mu = {mux, muy, muz};
  sc = {0.5 * std::log(sorted_vals[0]), 0.5 * std::log(sorted_vals[1]), 0.5 * std::log(sorted_vals[2])};
  q = {qtmp[0], qtmp[1], qtmp[2], qtmp[3]};
  op_lin = alpha_i + alpha_j - alpha_i * alpha_j;

  sh_out.resize(app_cols.size());
  for (size_t k = 0; k < app_cols.size(); ++k) {
    sh_out[k] = (wi * static_cast<double>((*app_cols[k])[i]) + wj * static_cast<double>((*app_cols[k])[j])) / W;
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
  if (targetCount <= 0) {
    return empty_table_like(data_table, 0);
  }
  if (N <= static_cast<size_t>(targetCount)) {
    return data_table.clone();
  }

  if (!has_required_columns(data_table)) {
    std::vector<uint32_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0u);
    sortByVisibility(data_table, indices);
    indices.resize(static_cast<size_t>(targetCount));
    return data_table.permuteRows(indices);
  }

  std::vector<std::string> appearance_names;
  collect_appearance_columns(data_table, appearance_names);

  const auto& opacity_col = data_table.getColumnByName("opacity").asVector<float>();
  std::vector<float> ops_sorted(N);
  for (size_t i = 0; i < N; ++i) {
    ops_sorted[i] = splat::sigmoid(opacity_col[i]);
  }
  std::sort(ops_sorted.begin(), ops_sorted.end());
  const float median = ops_sorted[N >> 1];
  const float prune_threshold = std::min(kOpacityPruneThreshold, median);

  std::vector<uint32_t> kept;
  kept.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    if (splat::sigmoid(opacity_col[i]) >= prune_threshold) {
      kept.push_back(static_cast<uint32_t>(i));
    }
  }

  std::unique_ptr<DataTable> current;
  if (kept.size() < N && kept.size() > static_cast<size_t>(targetCount)) {
    current = data_table.permuteRows(kept);
  } else {
    current = data_table.clone();
  }

  const std::vector<std::array<double, 3>> Z = dm::make_gaussian_samples(kMcSamples, 0);
  const bool profile = decimate_profile_enabled();
  size_t iteration = 0;

  while (current->getNumRows() > static_cast<size_t>(targetCount)) {
    ++iteration;
    const size_t n = current->getNumRows();
    const size_t k_eff = static_cast<size_t>(std::min(std::max(1, kKnnK), static_cast<int>(std::max<size_t>(1, n - 1))));
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
      LOG_INFO("decimate[%zu] start rows=%zu target=%d k=%zu", iteration, n, targetCount, k_eff);
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
    log_phase("knn");
    if (profile) {
      LOG_INFO("decimate[%zu] candidate_edges=%zu", iteration, edge_count);
    }

    if (edge_count == 0) {
      log_total();
      break;
    }

    std::vector<const std::vector<float>*> app_data;
    app_data.reserve(appearance_names.size());
    for (const std::string& name : appearance_names) {
      app_data.push_back(&current->getColumnByName(name).asVector<float>());
    }

    const size_t merges_needed = n - static_cast<size_t>(targetCount);
    std::vector<float> costs(edge_count);
    for (size_t e = 0; e < edge_count; ++e) {
      costs[e] = static_cast<float>(compute_edge_cost(edge_u[e], edge_v[e], cx, cy, cz, cache, Z, app_data));
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
                   app_data);

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

    current = std::move(new_table);
  }

  return current;
}

}  // namespace splat
