#include <splat/models/splatcloud.h>
#include <splat/visualization/splat_visualizer.h>
#include <vtkCamera.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkUnsignedCharArray.h>
#include <vtkWindowToImageFilter.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

struct VisibleBounds {
  int minX = 0;
  int maxX = -1;
  int minY = 0;
  int maxY = -1;

  bool valid() const { return maxX >= minX && maxY >= minY; }
  int width() const { return valid() ? (maxX - minX + 1) : 0; }
  int height() const { return valid() ? (maxY - minY + 1) : 0; }
};

VisibleBounds computeVisibleBounds(vtkImageData* image, int minX, int maxX, int minY, int maxY) {
  VisibleBounds bounds;
  int dims[3] = {0, 0, 0};
  image->GetDimensions(dims);

  auto* scalars = vtkUnsignedCharArray::SafeDownCast(image->GetPointData()->GetScalars());
  assert(scalars != nullptr);

  bool initialized = false;
  const int clampedMinX = std::max(0, minX);
  const int clampedMaxX = std::min(dims[0] - 1, maxX);
  const int clampedMinY = std::max(0, minY);
  const int clampedMaxY = std::min(dims[1] - 1, maxY);
  for (int y = clampedMinY; y <= clampedMaxY; ++y) {
    for (int x = clampedMinX; x <= clampedMaxX; ++x) {
      const vtkIdType index = static_cast<vtkIdType>(y) * dims[0] + x;
      unsigned char rgba[4] = {0, 0, 0, 0};
      scalars->GetTypedTuple(index, rgba);
      if (static_cast<int>(rgba[0]) + static_cast<int>(rgba[1]) + static_cast<int>(rgba[2]) < 160) {
        continue;
      }

      if (!initialized) {
        bounds.minX = bounds.maxX = x;
        bounds.minY = bounds.maxY = y;
        initialized = true;
      } else {
        bounds.minX = std::min(bounds.minX, x);
        bounds.maxX = std::max(bounds.maxX, x);
        bounds.minY = std::min(bounds.minY, y);
        bounds.maxY = std::max(bounds.maxY, y);
      }
    }
  }

  return bounds;
}

vtkSmartPointer<vtkImageData> renderReferenceFrame() {
  using namespace splat;

  SplatCloud table;
  const float quarterTurn = std::sqrt(0.5f);
  table.addColumn(Column{"x", std::vector<float>{-5.0f, 5.0f}});
  table.addColumn(Column{"y", std::vector<float>{0.0f, 0.0f}});
  table.addColumn(Column{"z", std::vector<float>{0.0f, 0.0f}});
  table.addColumn(Column{"f_dc_0", std::vector<float>{0.5f, 0.5f}});
  table.addColumn(Column{"f_dc_1", std::vector<float>{0.25f, 0.25f}});
  table.addColumn(Column{"f_dc_2", std::vector<float>{0.0f, 0.0f}});
  table.addColumn(Column{"opacity", std::vector<float>{4.0f, 4.0f}});
  table.addColumn(Column{"scale_0", std::vector<float>{std::log(2.5f), std::log(2.5f)}});
  table.addColumn(Column{"scale_1", std::vector<float>{std::log(0.35f), std::log(0.35f)}});
  table.addColumn(Column{"scale_2", std::vector<float>{std::log(0.35f), std::log(0.35f)}});
  table.addColumn(Column{"rot_0", std::vector<float>{1.0f, quarterTurn}});
  table.addColumn(Column{"rot_1", std::vector<float>{0.0f, 0.0f}});
  table.addColumn(Column{"rot_2", std::vector<float>{0.0f, 0.0f}});
  table.addColumn(Column{"rot_3", std::vector<float>{0.0f, quarterTurn}});

  SplatVisualizer visualizer("semantics");
  visualizer.getRenderWindow()->SetOffScreenRendering(1);
  visualizer.setWindowSize(256, 256);
  visualizer.setBackgroundColor(0.0, 0.0, 0.0);
  visualizer.setAxesEnabled(false);

  SplatRenderOptions options;
  options.sizeScale = 1.0f;
  options.minPointSize = 1.0f;
  options.maxPointSize = 128.0f;
  options.depthTest = false;
  options.depthWrite = false;

  const bool added = visualizer.addSplatCloud(table, "one", options);
  assert(added);

  auto* camera = visualizer.getRenderer()->GetActiveCamera();
  camera->SetPosition(0.0, 0.0, 24.0);
  camera->SetFocalPoint(0.0, 0.0, 0.0);
  camera->SetViewUp(0.0, 1.0, 0.0);
  visualizer.getRenderer()->ResetCameraClippingRange();
  visualizer.render();

  vtkNew<vtkWindowToImageFilter> capture;
  capture->SetInput(visualizer.getRenderWindow());
  capture->ReadFrontBufferOff();
  capture->Update();

  auto output = vtkSmartPointer<vtkImageData>::New();
  output->DeepCopy(capture->GetOutput());
  return output;
}

void expectNearClipSplatRemainsVisible() {
  using namespace splat;

  SplatCloud table;
  table.addColumn(Column{"x", std::vector<float>{0.0f}});
  table.addColumn(Column{"y", std::vector<float>{0.0f}});
  table.addColumn(Column{"z", std::vector<float>{0.7f}});
  table.addColumn(Column{"f_dc_0", std::vector<float>{1.0f}});
  table.addColumn(Column{"f_dc_1", std::vector<float>{1.0f}});
  table.addColumn(Column{"f_dc_2", std::vector<float>{1.0f}});
  table.addColumn(Column{"opacity", std::vector<float>{8.0f}});
  table.addColumn(Column{"scale_0", std::vector<float>{std::log(0.2f)}});
  table.addColumn(Column{"scale_1", std::vector<float>{std::log(0.2f)}});
  table.addColumn(Column{"scale_2", std::vector<float>{std::log(0.2f)}});
  table.addColumn(Column{"rot_0", std::vector<float>{1.0f}});
  table.addColumn(Column{"rot_1", std::vector<float>{0.0f}});
  table.addColumn(Column{"rot_2", std::vector<float>{0.0f}});
  table.addColumn(Column{"rot_3", std::vector<float>{0.0f}});

  SplatVisualizer visualizer("near-clip");
  visualizer.getRenderWindow()->SetOffScreenRendering(1);
  visualizer.setWindowSize(128, 128);
  visualizer.setBackgroundColor(0.0, 0.0, 0.0);
  visualizer.setAxesEnabled(false);

  SplatRenderOptions options;
  options.depthTest = false;
  options.depthWrite = false;
  const bool added = visualizer.addSplatCloud(table, "near", options);
  assert(added);

  auto* camera = visualizer.getRenderer()->GetActiveCamera();
  camera->SetPosition(0.0, 0.0, 1.0);
  camera->SetFocalPoint(0.0, 0.0, 0.0);
  camera->SetViewUp(0.0, 1.0, 0.0);
  camera->SetClippingRange(0.5, 10.0);
  visualizer.render();

  vtkNew<vtkWindowToImageFilter> capture;
  capture->SetInput(visualizer.getRenderWindow());
  capture->ReadFrontBufferOff();
  capture->Update();

  auto* image = capture->GetOutput();
  int dims[3] = {0, 0, 0};
  image->GetDimensions(dims);
  const VisibleBounds bounds = computeVisibleBounds(image, 0, dims[0] - 1, 0, dims[1] - 1);
  assert(bounds.valid());
}

void expectMissingColumnFailure() {
  using namespace splat;

  SplatCloud table;
  table.addColumn(Column{"x", std::vector<float>{0.0f}});
  table.addColumn(Column{"y", std::vector<float>{0.0f}});
  table.addColumn(Column{"z", std::vector<float>{0.0f}});
  table.addColumn(Column{"f_dc_0", std::vector<float>{0.5f}});
  table.addColumn(Column{"f_dc_1", std::vector<float>{0.25f}});
  table.addColumn(Column{"opacity", std::vector<float>{4.0f}});
  table.addColumn(Column{"scale_0", std::vector<float>{std::log(1.0f)}});
  table.addColumn(Column{"scale_1", std::vector<float>{std::log(1.0f)}});
  table.addColumn(Column{"scale_2", std::vector<float>{std::log(1.0f)}});
  table.addColumn(Column{"rot_0", std::vector<float>{1.0f}});
  table.addColumn(Column{"rot_1", std::vector<float>{0.0f}});
  table.addColumn(Column{"rot_2", std::vector<float>{0.0f}});
  table.addColumn(Column{"rot_3", std::vector<float>{0.0f}});

  SplatVisualizer visualizer("missing-column");

  bool threw = false;
  try {
    static_cast<void>(visualizer.addSplatCloud(table, "broken"));
  } catch (const std::runtime_error& error) {
    threw = true;
    const std::string message = error.what();
    assert(message.find("SplatVisualizer: missing required column: f_dc_2") != std::string::npos);
  }

  assert(threw);
}

}  // namespace

int main() {
  // This test suite exists because the TS renderer is known-good while the
  // current C++ path is known to produce no useful visible output. Any future
  // change that keeps compilation green but breaks decode/render semantics
  // should fail here first.
  auto referenceImage = renderReferenceFrame();
  int dims[3] = {0, 0, 0};
  referenceImage->GetDimensions(dims);

  const VisibleBounds horizontalBounds = computeVisibleBounds(referenceImage, 0, dims[0] / 2 - 1, 0, dims[1] - 1);
  assert(horizontalBounds.valid());
  const VisibleBounds verticalBounds = computeVisibleBounds(referenceImage, dims[0] / 2, dims[0] - 1, 0, dims[1] - 1);
  assert(verticalBounds.valid());

  assert(horizontalBounds.width() > horizontalBounds.height());
  assert(verticalBounds.height() > verticalBounds.width());
  expectNearClipSplatRemainsVisible();
  expectMissingColumnFailure();
  return 0;
}
