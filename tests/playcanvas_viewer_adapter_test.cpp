// Correct assert-based tests for playcanvas_viewer_adapter

#include <splat/models/splatcloud.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gsplat_data_adapter.h"

using playcanvas_viewer::adaptDataTableToGSplat;
using playcanvas_viewer::decodePlayCanvasColor;
using playcanvas_viewer::decodePlayCanvasOpacity;
using playcanvas_viewer::decodePlayCanvasScale;
using playcanvas_viewer::GSplatRenderData;
using playcanvas_viewer::normalizePlayCanvasRotation;
using playcanvas_viewer::Vec4f;
using splat::SplatCloud;

void test_decodePlayCanvasColor() { assert(std::abs(decodePlayCanvasColor(0) - 0.5f) < 1e-6f); }

void test_decodePlayCanvasOpacity() { assert(std::abs(decodePlayCanvasOpacity(0) - 0.5f) < 1e-6f); }

void test_decodePlayCanvasScale() { assert(std::abs(decodePlayCanvasScale(std::log(5.0f)) - 5.0f) < 1e-5f); }

void test_normalizePlayCanvasRotation() {
  // -2,0,0,0 should return identity {1,0,0,0}
  Vec4f out = normalizePlayCanvasRotation(-2, 0, 0, 0);
  assert(std::abs(out.x - 1.0f) < 1e-6f);
  assert(std::abs(out.y) < 1e-6f);
  assert(std::abs(out.z) < 1e-6f);
  assert(std::abs(out.w) < 1e-6f);
  // NaN,0,0,0 should return identity
  float nan = std::nanf("");
  Vec4f out2 = normalizePlayCanvasRotation(nan, 0, 0, 0);
  assert(std::abs(out2.x - 1.0f) < 1e-6f);
  assert(std::abs(out2.y) < 1e-6f);
  assert(std::abs(out2.z) < 1e-6f);
  assert(std::abs(out2.w) < 1e-6f);
}

void test_adaptDataTableToGSplat_one_row() {
  SplatCloud table;
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
  GSplatRenderData data = adaptDataTableToGSplat(table);
  assert(data.centers.size() == 1);
  assert(std::abs(data.centers[0].x - 1.0f) < 1e-6f);
  assert(std::abs(data.centers[0].y - 2.0f) < 1e-6f);
  assert(std::abs(data.centers[0].z - 3.0f) < 1e-6f);
  assert(std::abs(data.scales[0].x - 2.0f) < 1e-5f);
  assert(std::abs(data.scales[0].y - 3.0f) < 1e-5f);
  assert(std::abs(data.scales[0].z - 4.0f) < 1e-5f);
  assert(std::abs(data.colors[0].x - 0.5f) < 1e-6f);
  assert(std::abs(data.colors[0].y - 0.5f) < 1e-6f);
  assert(std::abs(data.colors[0].z - 0.5f) < 1e-6f);
  assert(std::abs(data.colors[0].w - 0.5f) < 1e-6f);  // alpha
  assert(data.sourceIndices.size() == 1);
  assert(data.sourceIndices[0] == 0);
  // Rotation should be identity
  assert(std::abs(data.rotations[0].x - 1.0f) < 1e-6f);
  assert(std::abs(data.rotations[0].y) < 1e-6f);
  assert(std::abs(data.rotations[0].z) < 1e-6f);
  assert(std::abs(data.rotations[0].w) < 1e-6f);
}

void test_adaptDataTableToGSplat_maxSplats() {
  SplatCloud table;
  // Add all required columns with two rows
  std::vector<std::string> names = {"x",       "y",       "z",       "rot_0",  "rot_1",  "rot_2",  "rot_3",
                                    "scale_0", "scale_1", "scale_2", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"};
  for (const auto& name : names) {
    table.addColumn({name, std::vector<float>{}});
  }
  for (const auto& name : names) {
    table.getColumnByName(name).asVector<float>().push_back(1.0f);
    table.getColumnByName(name).asVector<float>().push_back(2.0f);
  }
  GSplatRenderData data = adaptDataTableToGSplat(table, 1);  // maxSplats=1
  assert(data.centers.size() == 1);
}

void test_adaptDataTableToGSplat_missing_column() {
  SplatCloud table;
  // Only add some columns
  table.addColumn({"x", std::vector<float>{1.0f}});
  table.addColumn({"y", std::vector<float>{2.0f}});
  table.addColumn({"z", std::vector<float>{3.0f}});
  // Missing others
  bool threw = false;
  try {
    GSplatRenderData data = adaptDataTableToGSplat(table);
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()).find("missing required column") != std::string::npos;
  }
  assert(threw);
}

int main() {
  test_decodePlayCanvasColor();
  test_decodePlayCanvasOpacity();
  test_decodePlayCanvasScale();
  test_normalizePlayCanvasRotation();
  test_adaptDataTableToGSplat_one_row();
  test_adaptDataTableToGSplat_maxSplats();
  test_adaptDataTableToGSplat_missing_column();
  std::cout << "All PlaycanvasViewerAdapter tests passed.\n";
  return 0;
}
