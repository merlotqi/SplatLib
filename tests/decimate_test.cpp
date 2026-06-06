#include <splat/models/splatcloud.h>
#include <splat/op/decimate.h>

#include <cassert>
#include <cmath>
#include <memory>
#include <vector>

namespace {

std::vector<float> filled(size_t n, float value) { return std::vector<float>(n, value); }

void assertFiniteColumn(const splat::SplatCloud& table, const char* name) {
  const auto& values = table.getColumnByName(name).asVector<float>();
  for (float value : values) {
    assert(std::isfinite(value));
  }
}

}  // namespace

int main() {
  constexpr size_t n = 8;
  splat::SplatCloud table({
      {"x", std::vector<float>{0.0f, 0.01f, 1.0f, 1.01f, 2.0f, 2.01f, 3.0f, 3.01f}},
      {"y", std::vector<float>{0.0f, 0.01f, 0.0f, 0.01f, 0.0f, 0.01f, 0.0f, 0.01f}},
      {"z", filled(n, 0.0f)},
      {"opacity", filled(n, 4.0f)},
      {"scale_0", filled(n, -3.0f)},
      {"scale_1", filled(n, -3.0f)},
      {"scale_2", filled(n, -3.0f)},
      {"rot_0", filled(n, 1.0f)},
      {"rot_1", filled(n, 0.0f)},
      {"rot_2", filled(n, 0.0f)},
      {"rot_3", filled(n, 0.0f)},
      {"f_dc_0", filled(n, 0.2f)},
      {"f_dc_1", filled(n, 0.3f)},
      {"f_dc_2", filled(n, 0.4f)},
  });

  std::unique_ptr<splat::SplatCloud> simplified = splat::simplifyGaussians(table, 4);
  assert(simplified->getNumRows() == 4);

  for (const char* name : {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2",
                           "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"}) {
    assert(simplified->hasColumn(name));
    assertFiniteColumn(*simplified, name);
  }

  return 0;
}
