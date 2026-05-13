#include <assert.h>
#include <splat/models/data-table.h>
#include <splat/op/filter_visibility.h>

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
