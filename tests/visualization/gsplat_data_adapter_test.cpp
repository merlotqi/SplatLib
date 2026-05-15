#include <splat/models/data-table.h>
#include <splat/visualization/gsplat_data.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool nearlyEqual(float lhs, float rhs, float epsilon = 1e-5f) { return std::abs(lhs - rhs) <= epsilon; }

splat::DataTable makeOneRowTable() {
  splat::DataTable table;
  table.addColumn({"x", std::vector<float>{1.0f}});
  table.addColumn({"y", std::vector<float>{2.0f}});
  table.addColumn({"z", std::vector<float>{3.0f}});
  table.addColumn({"rot_0", std::vector<float>{-2.0f}});
  table.addColumn({"rot_1", std::vector<float>{0.0f}});
  table.addColumn({"rot_2", std::vector<float>{0.0f}});
  table.addColumn({"rot_3", std::vector<float>{0.0f}});
  table.addColumn({"scale_0", std::vector<float>{std::log(2.0f)}});
  table.addColumn({"scale_1", std::vector<float>{std::log(3.0f)}});
  table.addColumn({"scale_2", std::vector<float>{std::log(4.0f)}});
  table.addColumn({"f_dc_0", std::vector<float>{0.0f}});
  table.addColumn({"f_dc_1", std::vector<float>{0.0f}});
  table.addColumn({"f_dc_2", std::vector<float>{0.0f}});
  table.addColumn({"opacity", std::vector<float>{0.0f}});
  return table;
}

void testDecodeHelpers() {
  using namespace splat::visualization;
  require(nearlyEqual(decodePlayCanvasColor(0.0f), 0.5f), "DC color should decode through SH C0");
  require(nearlyEqual(decodePlayCanvasOpacity(0.0f), 0.5f), "opacity zero should decode to 0.5");
  require(nearlyEqual(decodePlayCanvasScale(std::log(5.0f)), 5.0f), "scale should decode with exp");
}

void testAdaptOneRow() {
  const auto table = makeOneRowTable();
  const auto data = splat::visualization::adaptDataTableToGSplatData(table);

  require(data.size() == 1, "one input row should produce one splat");
  require(nearlyEqual(data.centers[0].x, 1.0f), "center x should pass through");
  require(nearlyEqual(data.centers[0].y, 2.0f), "center y should pass through");
  require(nearlyEqual(data.centers[0].z, 3.0f), "center z should pass through");
  require(nearlyEqual(data.scales[0].x, 2.0f), "scale x should decode");
  require(nearlyEqual(data.scales[0].y, 3.0f), "scale y should decode");
  require(nearlyEqual(data.scales[0].z, 4.0f), "scale z should decode");
  require(nearlyEqual(data.colors[0].x, 0.5f), "red should decode");
  require(nearlyEqual(data.colors[0].y, 0.5f), "green should decode");
  require(nearlyEqual(data.colors[0].z, 0.5f), "blue should decode");
  require(nearlyEqual(data.colors[0].w, 0.5f), "alpha should decode");
  require(nearlyEqual(data.rotations[0].x, 1.0f), "negative w should normalize to positive identity");
  require(nearlyEqual(data.rotations[0].y, 0.0f), "rotation x should be zero");
  require(nearlyEqual(data.rotations[0].z, 0.0f), "rotation y should be zero");
  require(nearlyEqual(data.rotations[0].w, 0.0f), "rotation z should be zero");
  require(data.sourceIndices[0] == 0, "source index should be preserved");
}

void testMaxSplatsLimit() {
  splat::DataTable table;
  const std::vector<std::string> names = {"x",      "y",       "z",       "rot_0",  "rot_1",
                                          "rot_2",  "rot_3",   "scale_0", "scale_1", "scale_2",
                                          "f_dc_0", "f_dc_1",  "f_dc_2",  "opacity"};
  for (const auto& name : names) {
    table.addColumn({name, std::vector<float>{1.0f, 2.0f}});
  }

  const auto data = splat::visualization::adaptDataTableToGSplatData(table, 1);
  require(data.size() == 1, "maxSplats should limit output count");
}

void testMissingColumnThrows() {
  splat::DataTable table;
  table.addColumn({"x", std::vector<float>{1.0f}});
  table.addColumn({"y", std::vector<float>{2.0f}});
  table.addColumn({"z", std::vector<float>{3.0f}});

  bool threw = false;
  try {
    static_cast<void>(splat::visualization::adaptDataTableToGSplatData(table));
  } catch (const std::runtime_error& error) {
    threw = std::string(error.what()).find("missing required column") != std::string::npos;
  }
  require(threw, "missing required column should throw");
}

}  // namespace

int main() {
  testDecodeHelpers();
  testAdaptOneRow();
  testMaxSplatsLimit();
  testMissingColumnThrows();
  std::cout << "All GSplatData adapter tests passed.\n";
  return 0;
}
