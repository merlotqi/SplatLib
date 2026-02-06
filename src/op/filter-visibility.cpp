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

#pragma once

#include <assert.h>
#include <splat/models/data-table.h>

#include <algorithm>

namespace splat {

void sortByVisibility(const DataTable* dataTable, std::vector<unsigned int>& indices) {
  assert(dataTable);

  auto&& opacityCol = dataTable->getColumnByName("opacity");
  auto&& scale0Col = dataTable->getColumnByName("scale_0");
  auto&& scale1Col = dataTable->getColumnByName("scale_1");
  auto&& scale2Col = dataTable->getColumnByName("scale_2");

  if (indices.size() == 0) {
    return;
  }

  auto&& opacity = opacityCol.asSpan<float>();
  auto&& scale0 = scale0Col.asSpan<float>();
  auto&& scale1 = scale1Col.asSpan<float>();
  auto&& scale2 = scale2Col.asSpan<float>();

  // Compute visibility scores for each splat
  std::vector<float> scores(indices.size(), 0.0f);
  for (size_t i = 0; i < indices.size(); i++) {
    const auto ri = indices[i];

    // Convert logit opacity to linear using sigmoid
    const auto& logitOpacity = opacity[ri];
    const auto& linearOpacity = 1 / (1 + expf(-logitOpacity));

    // Convert log scales to linear and compute volume
    // volume = exp(scale_0) * exp(scale_1) * exp(scale_2) = exp(scale_0 + scale_1 + scale_2)
    const auto volume = expf(scale0[ri] + scale1[ri] + scale2[ri]);

    // Visibility score is opacity * volume
    scores[i] = linearOpacity * volume;
  }

  // Sort indices by score (descending - most visible first)
  std::vector<unsigned int> order(indices.size());
  for (size_t i = 0; i < order.size(); i++) {
    order[i] = i;
  }
  std::sort(order.begin(), order.end(), [&](unsigned int a, unsigned int b) { return scores[b] < scores[a]; });

  // Apply the sorted order to indices
  std::vector<unsigned int> tmpIndices = indices;
  for (size_t i = 0; i < indices.size(); i++) {
    indices[i] = tmpIndices[order[i]];
  }
}

}  // namespace splat
