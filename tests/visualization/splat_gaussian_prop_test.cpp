#include <splat/models/data-table.h>
#include "gsplat_gl_renderer.h"
#include "splat_gaussian_prop.h"

#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

splat::DataTable makeSingleSplatTable() {
  splat::DataTable table;
  table.addColumn({"x", std::vector<float>{0.0f}});
  table.addColumn({"y", std::vector<float>{0.0f}});
  table.addColumn({"z", std::vector<float>{0.0f}});
  table.addColumn({"f_dc_0", std::vector<float>{0.5f}});
  table.addColumn({"f_dc_1", std::vector<float>{0.25f}});
  table.addColumn({"f_dc_2", std::vector<float>{0.0f}});
  table.addColumn({"opacity", std::vector<float>{4.0f}});
  table.addColumn({"scale_0", std::vector<float>{std::log(1.0f)}});
  table.addColumn({"scale_1", std::vector<float>{std::log(1.0f)}});
  table.addColumn({"scale_2", std::vector<float>{std::log(1.0f)}});
  table.addColumn({"rot_0", std::vector<float>{1.0f}});
  table.addColumn({"rot_1", std::vector<float>{0.0f}});
  table.addColumn({"rot_2", std::vector<float>{0.0f}});
  table.addColumn({"rot_3", std::vector<float>{0.0f}});
  return table;
}

void test_public_prop_accepts_data_table_and_vtk_renderer() {
  auto prop = vtkSmartPointer<splat::SplatGaussianProp>::New();
  prop->SetInputData(makeSingleSplatTable());

  splat::SplatRenderOptions options;
  options.sortBackToFront = true;
  options.depthWrite = false;
  prop->SetRenderOptions(options);

  require(prop->GetSplatCount() == 1, "public prop should expose adapted splat count");
  require(prop->GetBounds() != nullptr, "public prop should expose VTK bounds after data upload");

  auto renderer = vtkSmartPointer<vtkRenderer>::New();
  renderer->AddViewProp(prop);
  require(renderer->HasViewProp(prop), "public prop should be accepted by vtkRenderer");
}

void test_public_render_options_match_shared_gl_defaults() {
  const splat::SplatRenderOptions publicOptions;
  const splat::visualization::GSplatGLRenderOptions glOptions;

  require(publicOptions.globalOpacity == glOptions.globalOpacity, "default opacity should match GL renderer");
  require(publicOptions.sizeScale == glOptions.sizeScale, "default size scale should match GL renderer");
  require(publicOptions.minPointSize == glOptions.minPixelSize, "default min point size should match GL renderer");
  require(publicOptions.maxPointSize == glOptions.maxPixelSize, "default max point size should match GL renderer");
  require(publicOptions.alphaDiscardThreshold == glOptions.alphaDiscardThreshold,
          "default alpha discard should match GL renderer");
  require(publicOptions.depthTest == glOptions.depthTest, "default depth test should match GL renderer");
  require(publicOptions.depthWrite == glOptions.depthWrite, "default depth write should match GL renderer");
  require(publicOptions.sortBackToFront == glOptions.sortBackToFront, "default sorting should match GL renderer");
  require(publicOptions.clampColors == glOptions.clampColors, "default color clamp should match GL renderer");
  require(publicOptions.freezeSortOrder == glOptions.freezeSortOrder, "default sort freeze should match GL renderer");
}

}  // namespace

int main() {
  test_public_render_options_match_shared_gl_defaults();
  test_public_prop_accepts_data_table_and_vtk_renderer();
  return 0;
}
