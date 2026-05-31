#include <splat/visualization/splat_visualizer.h>

#include <vtkCamera.h>
#include <vtkMatrix4x4.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

#include <cmath>

namespace {

bool nearlyEqual(double lhs, double rhs, double epsilon = 1e-6) { return std::abs(lhs - rhs) <= epsilon; }

bool matrixElementNearlyEqual(vtkMatrix4x4* matrix, int row, int column, double expected) {
  return nearlyEqual(matrix->GetElement(row, column), expected);
}

}  // namespace

int main() {
  splat::SplatVisualizer visualizer("smoke");
  visualizer.setWindowSize(320, 240);
  visualizer.setAxesEnabled(false);

  splat::SplatRenderOptions options;
  options.sortBackToFront = true;
  options.clampColors = true;

  auto renderer = vtkSmartPointer<vtkRenderer>::New();
  auto renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  splat::SplatVisualizer embedded(renderer, renderWindow, interactor, "embedded");
  if (embedded.getRenderer() != renderer || embedded.getRenderWindow() != renderWindow || embedded.getInteractor() != interactor) {
    return 2;
  }

  embedded.setWindowSize(1280, 720);
  const double bounds[6] = {-10.0, 10.0, -10.0, 10.0, -10.0, 10.0};
  embedded.resetCameraToBounds(bounds);
  auto* camera = embedded.getRenderer()->GetActiveCamera();
  double position[3] = {0.0, 0.0, 0.0};
  double focalPoint[3] = {0.0, 0.0, 0.0};
  double viewUp[3] = {0.0, 0.0, 0.0};
  double clippingRange[2] = {0.0, 0.0};
  camera->GetPosition(position);
  camera->GetFocalPoint(focalPoint);
  camera->GetViewUp(viewUp);
  camera->GetClippingRange(clippingRange);
  const double radius = std::sqrt(10.0 * 10.0 + 10.0 * 10.0 + 10.0 * 10.0);
  const double expectedDistance = radius * 3.0;
  if (!nearlyEqual(camera->GetViewAngle(), 60.0) || !nearlyEqual(viewUp[0], 0.0) ||
      !nearlyEqual(viewUp[1], -1.0) || !nearlyEqual(viewUp[2], 0.0) || !nearlyEqual(focalPoint[0], 0.0) ||
      !nearlyEqual(focalPoint[1], 0.0) || !nearlyEqual(focalPoint[2], 0.0) ||
      !nearlyEqual(position[2], expectedDistance) || !nearlyEqual(clippingRange[0], radius * 0.0005) ||
      !nearlyEqual(clippingRange[1], radius * 32.0)) {
    return 3;
  }
  auto* viewMatrix = camera->GetViewTransformMatrix();
  if (!matrixElementNearlyEqual(viewMatrix, 0, 0, -1.0) || !matrixElementNearlyEqual(viewMatrix, 1, 1, -1.0) ||
      !matrixElementNearlyEqual(viewMatrix, 2, 2, 1.0) || !matrixElementNearlyEqual(viewMatrix, 2, 3, -expectedDistance)) {
    return 4;
  }
  auto* projectionMatrix = camera->GetProjectionTransformMatrix(1280.0 / 720.0, clippingRange[0], clippingRange[1]);
  const double tanHalfFov = std::tan((60.0 * 3.14159265358979323846 / 180.0) * 0.5);
  if (!matrixElementNearlyEqual(projectionMatrix, 0, 0, 1.0 / ((1280.0 / 720.0) * tanHalfFov)) ||
      !matrixElementNearlyEqual(projectionMatrix, 1, 1, 1.0 / tanHalfFov) ||
      !matrixElementNearlyEqual(projectionMatrix, 3, 2, -1.0)) {
    return 5;
  }
  return visualizer.wasStopped() || embedded.wasStopped() ? 1 : 0;
}
