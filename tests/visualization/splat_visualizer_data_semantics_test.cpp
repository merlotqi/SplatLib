#include <splat/models/data-table.h>
#include <splat/visualization/splat_visualizer.h>

#include <vtkCamera.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkUnsignedCharArray.h>
#include <vtkWindowToImageFilter.h>

#include <cassert>
#include <cmath>
#include <vector>

int main() {
  using namespace splat;

  // This test suite exists because the TS renderer is known-good while the
  // current C++ path is known to produce no useful visible output. Any future
  // change that keeps compilation green but breaks decode/render semantics
  // should fail here first.
  DataTable table;
  table.addColumn(Column{"x", std::vector<float>{0.0f}});
  table.addColumn(Column{"y", std::vector<float>{0.0f}});
  table.addColumn(Column{"z", std::vector<float>{0.0f}});
  table.addColumn(Column{"f_dc_0", std::vector<float>{0.5f}});
  table.addColumn(Column{"f_dc_1", std::vector<float>{0.25f}});
  table.addColumn(Column{"f_dc_2", std::vector<float>{0.0f}});
  table.addColumn(Column{"opacity", std::vector<float>{4.0f}});
  table.addColumn(Column{"scale_0", std::vector<float>{std::log(20.0f)}});
  table.addColumn(Column{"scale_1", std::vector<float>{std::log(20.0f)}});
  table.addColumn(Column{"scale_2", std::vector<float>{std::log(20.0f)}});
  table.addColumn(Column{"rot_0", std::vector<float>{1.0f}});
  table.addColumn(Column{"rot_1", std::vector<float>{0.0f}});
  table.addColumn(Column{"rot_2", std::vector<float>{0.0f}});
  table.addColumn(Column{"rot_3", std::vector<float>{0.0f}});

  SplatVisualizer visualizer("semantics");
  visualizer.getRenderWindow()->SetOffScreenRendering(1);
  visualizer.setWindowSize(256, 256);
  visualizer.setBackgroundColor(0.0, 0.0, 0.0);
  visualizer.setAxesEnabled(false);
  SplatRenderOptions options;
  options.sizeScale = 10.0f;
  options.minPointSize = 16.0f;
  options.maxPointSize = 256.0f;
  options.depthTest = false;
  options.depthWrite = false;
  const bool added = visualizer.addSplatCloud(table, "one", options);
  assert(added);
  assert(visualizer.contains("one"));
  assert(visualizer.getSplatCount("one") == 1);

  auto* camera = visualizer.getRenderer()->GetActiveCamera();
  camera->SetPosition(0.0, 0.0, 20.0);
  camera->SetFocalPoint(0.0, 0.0, 0.0);
  camera->SetViewUp(0.0, 1.0, 0.0);
  visualizer.getRenderer()->ResetCameraClippingRange();
  visualizer.render();

  bool foundVisiblePixel = false;
  vtkNew<vtkWindowToImageFilter> capture;
  capture->SetInput(visualizer.getRenderWindow());
  capture->ReadFrontBufferOff();
  capture->Update();

  auto* image = capture->GetOutput();
  auto* scalars = vtkUnsignedCharArray::SafeDownCast(image->GetPointData()->GetScalars());
  assert(scalars != nullptr);

  const auto pixelCount = scalars->GetNumberOfTuples();
  for (vtkIdType i = 0; i < pixelCount; ++i) {
    unsigned char rgba[4] = {0, 0, 0, 0};
    scalars->GetTypedTuple(i, rgba);
    if (rgba[0] != 0 || rgba[1] != 0 || rgba[2] != 0) {
      foundVisiblePixel = true;
      break;
    }
  }
  assert(foundVisiblePixel);
  return 0;
}
