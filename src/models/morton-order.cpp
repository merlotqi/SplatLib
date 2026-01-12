/***********************************************************************************
 *
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
 *
 ***********************************************************************************/

#include <splat/models/morton-order.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>

namespace splat {
static uint32_t part1By2(uint32_t x) {
  x &= 0x000003ff;                   // x = ---- ---- ---- ---- ---- --98 7654 3210
  x = (x ^ (x << 16)) & 0xff0000ff;  // x = ---- --98 ---- ---- ---- ---- 7654 3210
  x = (x ^ (x << 8)) & 0x0300f00f;   // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  x = (x ^ (x << 4)) & 0x030c30c3;   // x = ---- --98 ---- 76-- 54-- ---- 32-- 10--
  x = (x ^ (x << 2)) & 0x09249249;   // x = ---- 9--8 --7--6 --5--4 --3--2 --1--0
  return x;
}

static uint32_t encodeMorton3(uint32_t x, uint32_t y, uint32_t z) {
  return (part1By2(z) << 2) + (part1By2(y) << 1) + part1By2(x);
}

void sortMortonOrder(const DataTable* dataTable, absl::Span<uint32_t> indices) {
  if (indices.empty()) return;

  const auto& cx = dataTable->getColumnByName("x").asSpan<float>();
  const auto& cy = dataTable->getColumnByName("y").asSpan<float>();
  const auto& cz = dataTable->getColumnByName("z").asSpan<float>();

  float mx = std::numeric_limits<float>::max();
  float my = mx, mz = mx;
  float Mx = -mx, My = Mx, Mz = Mx;

  for (uint32_t ri : indices) {
    float x = cx[ri], y = cy[ri], z = cz[ri];
    if (x < mx) mx = x;
    if (x > Mx) Mx = x;
    if (y < my) my = y;
    if (y > My) My = y;
    if (z < mz) mz = z;
    if (z > Mz) Mz = z;
  }

  float xlen = Mx - mx;
  float ylen = My - my;
  float zlen = Mz - mz;

  if (!std::isfinite(xlen) || !std::isfinite(ylen) || !std::isfinite(zlen)) return;
  if (xlen == 0 && ylen == 0 && zlen == 0) return;

  float xmul = (xlen == 0.0f) ? 0.0f : 1024.0f / xlen;
  float ymul = (ylen == 0.0f) ? 0.0f : 1024.0f / ylen;
  float zmul = (zlen == 0.0f) ? 0.0f : 1024.0f / zlen;

  std::vector<uint32_t> mortonCodes(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    uint32_t ri = indices[i];
    uint32_t ix = std::min(1023u, static_cast<uint32_t>(std::max(0.0f, (cx[ri] - mx) * xmul)));
    uint32_t iy = std::min(1023u, static_cast<uint32_t>(std::max(0.0f, (cy[ri] - my) * ymul)));
    uint32_t iz = std::min(1023u, static_cast<uint32_t>(std::max(0.0f, (cz[ri] - mz) * zmul)));
    mortonCodes[i] = encodeMorton3(ix, iy, iz);
  }

  std::vector<uint32_t> order(indices.size());
  std::iota(order.begin(), order.end(), 0);

  std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return mortonCodes[a] < mortonCodes[b]; });

  std::vector<uint32_t> tmpIndices(indices.begin(), indices.end());
  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = tmpIndices[order[i]];
  }

  size_t start = 0;
  while (start < indices.size()) {
    size_t end = start + 1;
    while (end < indices.size() && mortonCodes[order[end]] == mortonCodes[order[start]]) {
      ++end;
    }

    if (end - start > 256) {
      sortMortonOrder(dataTable, indices.subspan(start, end - start));
    }
    start = end;
  }
}

}  // namespace splat
