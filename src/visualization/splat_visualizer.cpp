#include <splat/models/data-table.h>
#include <splat/spatial/gaussian_aabb.h>
#include <splat/visualization/gsplat_data.h>
#include <splat/visualization/gsplat_gl_renderer.h>
#include <splat/visualization/splat_visualizer.h>
#include <vtkAxesActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCommand.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkOpenGLCamera.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkOpenGLState.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkProp3D.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkViewport.h>
#include <vtkWindow.h>
#include <vtk_glew.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace splat {
namespace {

constexpr const char* kPositionColumns[3] = {"x", "y", "z"};
constexpr const char* kColorColumns[3] = {"f_dc_0", "f_dc_1", "f_dc_2"};
constexpr const char* kScaleColumns[3] = {"scale_0", "scale_1", "scale_2"};
constexpr const char* kRotationColumns[4] = {"rot_0", "rot_1", "rot_2", "rot_3"};
constexpr const char* kRequiredRenderColumns[] = {"x",      "y",       "z",       "f_dc_0", "f_dc_1",
                                                  "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
                                                  "rot_0",  "rot_1",   "rot_2",   "rot_3"};
constexpr int kCornerAttribLocation = 0;
constexpr int kPositionAttribLocation = 1;
constexpr int kColorAttribLocation = 2;
constexpr int kScaleAttribLocation = 3;
constexpr int kOpacityAttribLocation = 4;
constexpr int kRotationAttribLocation = 5;
constexpr std::array<float, 8> kQuadCorners = {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

std::shared_ptr<const DataTable> cloneShared(const DataTable& dataTable) {
  auto clone = dataTable.clone();
  return std::shared_ptr<const DataTable>(clone.release());
}

void requireRenderColumns(const DataTable& dataTable) {
  for (const char* name : kRequiredRenderColumns) {
    if (!dataTable.hasColumn(name)) {
      throw std::runtime_error(std::string("SplatVisualizer: missing required column: ") + name);
    }
  }
}

std::vector<float> extractScalarColumn(const DataTable& dataTable, const std::string& name) {
  const auto& column = dataTable.getColumnByName(name);
  std::vector<float> values(column.length());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = column.getValue<float>(i);
  }
  return values;
}

std::vector<float> extractVec3Columns(const DataTable& dataTable, const char* const (&names)[3]) {
  const auto& c0 = dataTable.getColumnByName(names[0]);
  const auto& c1 = dataTable.getColumnByName(names[1]);
  const auto& c2 = dataTable.getColumnByName(names[2]);
  const size_t count = dataTable.getNumRows();

  std::vector<float> values(count * 3);
  for (size_t i = 0; i < count; ++i) {
    const size_t base = i * 3;
    values[base + 0] = c0.getValue<float>(i);
    values[base + 1] = c1.getValue<float>(i);
    values[base + 2] = c2.getValue<float>(i);
  }
  return values;
}

std::vector<float> extractQuaternionColumns(const DataTable& dataTable, const char* const (&names)[4]) {
  const auto& c0 = dataTable.getColumnByName(names[0]);
  const auto& c1 = dataTable.getColumnByName(names[1]);
  const auto& c2 = dataTable.getColumnByName(names[2]);
  const auto& c3 = dataTable.getColumnByName(names[3]);
  const size_t count = dataTable.getNumRows();

  std::vector<float> values(count * 4);
  for (size_t i = 0; i < count; ++i) {
    const size_t base = i * 4;
    float w = c0.getValue<float>(i);
    float x = c1.getValue<float>(i);
    float y = c2.getValue<float>(i);
    float z = c3.getValue<float>(i);
    const float norm = std::sqrt(w * w + x * x + y * y + z * z);

    if (norm > 1e-8f && std::isfinite(norm)) {
      const float invNorm = 1.0f / norm;
      values[base + 0] = w * invNorm;
      values[base + 1] = x * invNorm;
      values[base + 2] = y * invNorm;
      values[base + 3] = z * invNorm;
    } else {
      values[base + 0] = 1.0f;
      values[base + 1] = 0.0f;
      values[base + 2] = 0.0f;
      values[base + 3] = 0.0f;
    }
  }
  return values;
}

std::array<double, 6> computeBoundsFromPositions(const std::vector<float>& positions) {
  if (positions.empty()) {
    return {1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
  }

  std::array<double, 6> bounds = {
      std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
  };

  for (size_t i = 0; i < positions.size(); i += 3) {
    bounds[0] = std::min(bounds[0], static_cast<double>(positions[i + 0]));
    bounds[1] = std::max(bounds[1], static_cast<double>(positions[i + 0]));
    bounds[2] = std::min(bounds[2], static_cast<double>(positions[i + 1]));
    bounds[3] = std::max(bounds[3], static_cast<double>(positions[i + 1]));
    bounds[4] = std::min(bounds[4], static_cast<double>(positions[i + 2]));
    bounds[5] = std::max(bounds[5], static_cast<double>(positions[i + 2]));
  }

  return bounds;
}

bool isValidBounds(const std::array<double, 6>& bounds) {
  return std::isfinite(bounds[0]) && std::isfinite(bounds[1]) && std::isfinite(bounds[2]) && std::isfinite(bounds[3]) &&
         std::isfinite(bounds[4]) && std::isfinite(bounds[5]) && bounds[0] <= bounds[1] && bounds[2] <= bounds[3] &&
         bounds[4] <= bounds[5];
}

std::array<double, 6> computeSplatBounds(const DataTable& dataTable, const std::vector<float>& positions) {
  const auto extents = computeGaussianExtents(&dataTable);
  const auto& minBound = extents.sceneBounds.min;
  const auto& maxBound = extents.sceneBounds.max;

  std::array<double, 6> bounds = {
      static_cast<double>(minBound.x()), static_cast<double>(maxBound.x()), static_cast<double>(minBound.y()),
      static_cast<double>(maxBound.y()), static_cast<double>(minBound.z()), static_cast<double>(maxBound.z()),
  };

  if (isValidBounds(bounds)) {
    return bounds;
  }

  return computeBoundsFromPositions(positions);
}

std::array<double, 6> computeBoundsFromGSplatData(const splat::visualization::GSplatData& data) {
  if (data.empty()) {
    return {1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
  }

  std::array<double, 6> bounds = {
      std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
      std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
  };

  for (const auto& center : data.centers) {
    bounds[0] = std::min(bounds[0], static_cast<double>(center.x));
    bounds[1] = std::max(bounds[1], static_cast<double>(center.x));
    bounds[2] = std::min(bounds[2], static_cast<double>(center.y));
    bounds[3] = std::max(bounds[3], static_cast<double>(center.y));
    bounds[4] = std::min(bounds[4], static_cast<double>(center.z));
    bounds[5] = std::max(bounds[5], static_cast<double>(center.z));
  }

  return bounds;
}

std::array<double, 6> computeSplatBounds(const DataTable& dataTable, const splat::visualization::GSplatData& data) {
  const auto extents = computeGaussianExtents(&dataTable);
  const auto& minBound = extents.sceneBounds.min;
  const auto& maxBound = extents.sceneBounds.max;

  std::array<double, 6> bounds = {
      static_cast<double>(minBound.x()), static_cast<double>(maxBound.x()), static_cast<double>(minBound.y()),
      static_cast<double>(maxBound.y()), static_cast<double>(minBound.z()), static_cast<double>(maxBound.z()),
  };

  if (isValidBounds(bounds)) {
    return bounds;
  }

  return computeBoundsFromGSplatData(data);
}

Eigen::Matrix4f copyMatrixToEigen(const vtkMatrix4x4* matrix) {
  Eigen::Matrix4f out = Eigen::Matrix4f::Identity();
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      out(row, col) = static_cast<float>(matrix->GetElement(row, col));
    }
  }
  return out;
}

KeyModifier makeModifiers(vtkRenderWindowInteractor* interactor) {
  KeyModifier modifiers = KeyModifier::None;
  if (interactor->GetShiftKey()) {
    modifiers = modifiers | KeyModifier::Shift;
  }
  if (interactor->GetControlKey()) {
    modifiers = modifiers | KeyModifier::Control;
  }
  if (interactor->GetAltKey()) {
    modifiers = modifiers | KeyModifier::Alt;
  }
  return modifiers;
}

MouseButton makeMouseButton(unsigned long eventId) {
  switch (eventId) {
    case vtkCommand::LeftButtonPressEvent:
    case vtkCommand::LeftButtonReleaseEvent:
    case vtkCommand::LeftButtonDoubleClickEvent:
      return MouseButton::Left;
    case vtkCommand::MiddleButtonPressEvent:
    case vtkCommand::MiddleButtonReleaseEvent:
    case vtkCommand::MiddleButtonDoubleClickEvent:
      return MouseButton::Middle;
    case vtkCommand::RightButtonPressEvent:
    case vtkCommand::RightButtonReleaseEvent:
    case vtkCommand::RightButtonDoubleClickEvent:
      return MouseButton::Right;
    case vtkCommand::FourthButtonPressEvent:
    case vtkCommand::FourthButtonReleaseEvent:
      return MouseButton::Button4;
    case vtkCommand::FifthButtonPressEvent:
    case vtkCommand::FifthButtonReleaseEvent:
      return MouseButton::Button5;
    default:
      return MouseButton::None;
  }
}

MouseAction makeMouseAction(unsigned long eventId) {
  switch (eventId) {
    case vtkCommand::MouseMoveEvent:
      return MouseAction::Move;
    case vtkCommand::LeftButtonPressEvent:
    case vtkCommand::MiddleButtonPressEvent:
    case vtkCommand::RightButtonPressEvent:
    case vtkCommand::FourthButtonPressEvent:
    case vtkCommand::FifthButtonPressEvent:
      return MouseAction::Press;
    case vtkCommand::LeftButtonReleaseEvent:
    case vtkCommand::MiddleButtonReleaseEvent:
    case vtkCommand::RightButtonReleaseEvent:
    case vtkCommand::FourthButtonReleaseEvent:
    case vtkCommand::FifthButtonReleaseEvent:
      return MouseAction::Release;
    case vtkCommand::LeftButtonDoubleClickEvent:
    case vtkCommand::MiddleButtonDoubleClickEvent:
    case vtkCommand::RightButtonDoubleClickEvent:
      return MouseAction::DoubleClick;
    case vtkCommand::MouseWheelForwardEvent:
    case vtkCommand::MouseWheelBackwardEvent:
    case vtkCommand::MouseWheelLeftEvent:
    case vtkCommand::MouseWheelRightEvent:
      return MouseAction::Wheel;
    default:
      return MouseAction::Move;
  }
}

int makeWheelDelta(unsigned long eventId) {
  switch (eventId) {
    case vtkCommand::MouseWheelForwardEvent:
    case vtkCommand::MouseWheelRightEvent:
      return 1;
    case vtkCommand::MouseWheelBackwardEvent:
    case vtkCommand::MouseWheelLeftEvent:
      return -1;
    default:
      return 0;
  }
}

void transformPoint(const double matrix[16], const std::array<double, 3>& point, double out[3]) {
  out[0] = matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3];
  out[1] = matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7];
  out[2] = matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11];
}

class NativeSplatProp : public vtkProp3D {
 public:
  static NativeSplatProp* New();
  vtkTypeMacro(NativeSplatProp, vtkProp3D);

  void SetInputData(std::shared_ptr<const DataTable> dataTable) {
    this->InputDataTable = std::move(dataTable);
    this->RebuildCpuCache();
    this->Modified();
  }

  void SetRenderOptions(const SplatRenderOptions& options) {
    this->RenderOptions = options;
    this->SetVisibility(options.visible ? 1 : 0);
    this->Modified();
  }

  size_t GetSplatCount() const noexcept { return this->RenderData.size(); }

  double* GetBounds() override {
    if (!this->BoundsValid) {
      return nullptr;
    }

    const double localBounds[6] = {
        this->LocalBounds[0], this->LocalBounds[1], this->LocalBounds[2],
        this->LocalBounds[3], this->LocalBounds[4], this->LocalBounds[5],
    };

    double modelMatrix[16];
    this->GetMatrix(modelMatrix);

    const std::array<double, 8 * 3> corners = {
        localBounds[0], localBounds[2], localBounds[4], localBounds[1], localBounds[2], localBounds[4],
        localBounds[0], localBounds[3], localBounds[4], localBounds[1], localBounds[3], localBounds[4],
        localBounds[0], localBounds[2], localBounds[5], localBounds[1], localBounds[2], localBounds[5],
        localBounds[0], localBounds[3], localBounds[5], localBounds[1], localBounds[3], localBounds[5],
    };

    this->WorldBounds = {
        std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
    };

    for (size_t i = 0; i < corners.size(); i += 3) {
      double transformed[3];
      transformPoint(modelMatrix, {corners[i + 0], corners[i + 1], corners[i + 2]}, transformed);
      this->WorldBounds[0] = std::min(this->WorldBounds[0], transformed[0]);
      this->WorldBounds[1] = std::max(this->WorldBounds[1], transformed[0]);
      this->WorldBounds[2] = std::min(this->WorldBounds[2], transformed[1]);
      this->WorldBounds[3] = std::max(this->WorldBounds[3], transformed[1]);
      this->WorldBounds[4] = std::min(this->WorldBounds[4], transformed[2]);
      this->WorldBounds[5] = std::max(this->WorldBounds[5], transformed[2]);
    }

    return this->WorldBounds.data();
  }

  int RenderOpaqueGeometry(vtkViewport* viewport) override { return this->RenderSplatGeometry(viewport); }

  int RenderTranslucentPolygonalGeometry(vtkViewport*) override { return 0; }

  vtkTypeBool HasTranslucentPolygonalGeometry() override { return 0; }

  void ReleaseGraphicsResources(vtkWindow* window) override {
    auto* openGLWindow = vtkOpenGLRenderWindow::SafeDownCast(window);
    if (openGLWindow == nullptr) {
      return;
    }

    openGLWindow->MakeCurrent();

    this->Renderer.releaseGraphicsResources();
  }

 protected:
  NativeSplatProp() = default;
  ~NativeSplatProp() override = default;

 private:
  void ResetCpuCache() {
    this->RenderData = {};
    this->Renderer.setData(this->RenderData);
    this->BoundsValid = false;
    this->LocalBounds = {1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
    this->WorldBounds = this->LocalBounds;
  }

  int RenderSplatGeometry(vtkViewport* viewport) {
    if (!this->GetVisibility() || this->RenderData.empty()) {
      return 0;
    }

    auto* renderer = vtkRenderer::SafeDownCast(viewport);
    if (renderer == nullptr) {
      return 0;
    }

    auto* openGLWindow = vtkOpenGLRenderWindow::SafeDownCast(renderer->GetRenderWindow());
    if (openGLWindow == nullptr) {
      throw std::runtime_error("SplatVisualizer requires a vtkOpenGLRenderWindow.");
    }

    openGLWindow->MakeCurrent();
    auto* state = openGLWindow->GetState();
    state->Push();

    auto* camera = renderer->GetActiveCamera();
    auto* openGLCamera = vtkOpenGLCamera::SafeDownCast(camera);
    if (openGLCamera == nullptr) {
      throw std::runtime_error("SplatVisualizer requires a vtkOpenGLCamera.");
    }

    vtkMatrix3x3* normalMatrix = nullptr;
    vtkMatrix4x4* viewToClipMatrix = nullptr;
    vtkMatrix4x4* worldToClipMatrix = nullptr;
    vtkMatrix4x4* unusedWorldToViewMatrix = nullptr;
    openGLCamera->GetKeyMatrices(renderer, unusedWorldToViewMatrix, normalMatrix, viewToClipMatrix, worldToClipMatrix);

    vtkNew<vtkMatrix4x4> modelMatrix;
    vtkNew<vtkMatrix4x4> modelViewMatrix;
    vtkNew<vtkMatrix4x4> modelClipMatrix;
    this->GetMatrix(modelMatrix);
    vtkMatrix4x4::Multiply4x4(camera->GetViewTransformMatrix(), modelMatrix, modelViewMatrix);

    const int* renderWindowSize = openGLWindow->GetSize();
    double clippingRange[2] = {0.001, 1000.0};
    camera->GetClippingRange(clippingRange);
    double cameraPosition[3] = {0.0, 0.0, 0.0};
    double cameraForward[3] = {0.0, 0.0, -1.0};
    camera->GetPosition(cameraPosition);
    camera->GetDirectionOfProjection(cameraForward);

    splat::visualization::GSplatGLFrameState frame;
    frame.view = copyMatrixToEigen(modelViewMatrix);
    frame.projection = copyMatrixToEigen(viewToClipMatrix);
    frame.cameraPosition = Eigen::Vector3f(static_cast<float>(cameraPosition[0]), static_cast<float>(cameraPosition[1]),
                                           static_cast<float>(cameraPosition[2]));
    frame.cameraForward = Eigen::Vector3f(static_cast<float>(cameraForward[0]), static_cast<float>(cameraForward[1]),
                                          static_cast<float>(cameraForward[2]));
    frame.width = std::max(renderWindowSize[0], 1);
    frame.height = std::max(renderWindowSize[1], 1);
    frame.nearPlane = static_cast<float>(clippingRange[0]);
    frame.farPlane = static_cast<float>(clippingRange[1]);

    splat::visualization::GSplatGLRenderOptions options;
    options.globalOpacity = this->RenderOptions.globalOpacity;
    options.sizeScale = this->RenderOptions.sizeScale;
    options.minPixelSize = this->RenderOptions.minPointSize;
    options.maxPixelSize = this->RenderOptions.maxPointSize;
    options.alphaDiscardThreshold = this->RenderOptions.alphaDiscardThreshold;
    options.sortBackToFront = this->RenderOptions.sortBackToFront;
    options.depthTest = this->RenderOptions.depthTest;
    options.depthWrite = this->RenderOptions.depthWrite;
    options.clampColors = this->RenderOptions.clampColors;

    this->Renderer.render(frame, options);
    state->Pop();
    return 1;
  }


  void RebuildCpuCache() {
    this->ResetCpuCache();

    if (!this->InputDataTable || this->InputDataTable->getNumRows() == 0) {
      return;
    }

    try {
      this->RenderData = splat::visualization::adaptDataTableToGSplatData(*this->InputDataTable);
    } catch (const std::runtime_error& error) {
      std::string message = error.what();
      const std::string prefix = "GSplatData: ";
      if (message.rfind(prefix, 0) == 0) {
        message.replace(0, prefix.size(), "SplatVisualizer: ");
      }
      throw std::runtime_error(message);
    }

    if (this->RenderData.empty()) {
      throw std::runtime_error("SplatVisualizer: upload produced zero splats");
    }

    this->Renderer.setData(this->RenderData);
    this->LocalBounds = computeSplatBounds(*this->InputDataTable, this->RenderData);
    this->BoundsValid = isValidBounds(this->LocalBounds);
  }

  std::shared_ptr<const DataTable> InputDataTable;
  SplatRenderOptions RenderOptions;
  splat::visualization::GSplatData RenderData;
  splat::visualization::GSplatGLRenderer Renderer;
  bool BoundsValid{false};
  std::array<double, 6> LocalBounds{1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
  std::array<double, 6> WorldBounds{1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
};

vtkStandardNewMacro(NativeSplatProp);

struct CloudEntry {
  std::shared_ptr<const DataTable> dataTable;
  vtkSmartPointer<NativeSplatProp> prop;
  SplatRenderOptions options;
};

}  // namespace

class SplatVisualizer::Impl {
 public:
  explicit Impl(std::string name)
      : windowName(std::move(name)),
        renderer(vtkSmartPointer<vtkRenderer>::New()),
        renderWindow(vtkSmartPointer<vtkRenderWindow>::New()),
        interactor(vtkSmartPointer<vtkRenderWindowInteractor>::New()),
        interactorStyle(vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New()),
        axesActor(vtkSmartPointer<vtkAxesActor>::New()),
        axesWidget(vtkSmartPointer<vtkOrientationMarkerWidget>::New()),
        keyObserver(vtkSmartPointer<vtkCallbackCommand>::New()),
        mouseObserver(vtkSmartPointer<vtkCallbackCommand>::New()),
        exitObserver(vtkSmartPointer<vtkCallbackCommand>::New()) {
    this->renderer->SetBackground(this->backgroundColor[0], this->backgroundColor[1], this->backgroundColor[2]);
    this->renderWindow->AddRenderer(this->renderer);
    this->renderWindow->SetWindowName(this->windowName.c_str());
    this->renderWindow->SetSize(1280, 720);
    this->renderWindow->SetMultiSamples(0);

    this->interactor->SetInteractorStyle(this->interactorStyle);
    this->interactor->SetRenderWindow(this->renderWindow);

    this->axesActor->SetTotalLength(this->axesLength, this->axesLength, this->axesLength);
    this->axesWidget->SetOrientationMarker(this->axesActor);
    this->axesWidget->SetInteractor(this->interactor);
    this->axesWidget->SetViewport(0.0, 0.0, 0.18, 0.18);
    this->axesWidget->InteractiveOff();
    this->axesWidget->SetEnabled(this->axesEnabled ? 1 : 0);

    this->keyObserver->SetClientData(this);
    this->keyObserver->SetCallback(&Impl::HandleKeyEvent);
    this->interactor->AddObserver(vtkCommand::KeyPressEvent, this->keyObserver);
    this->interactor->AddObserver(vtkCommand::KeyReleaseEvent, this->keyObserver);

    this->mouseObserver->SetClientData(this);
    this->mouseObserver->SetCallback(&Impl::HandleMouseEvent);
    this->interactor->AddObserver(vtkCommand::MouseMoveEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::LeftButtonPressEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::LeftButtonReleaseEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::MiddleButtonPressEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::MiddleButtonReleaseEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::RightButtonPressEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::RightButtonReleaseEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::FourthButtonPressEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::FourthButtonReleaseEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::FifthButtonPressEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::FifthButtonReleaseEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::LeftButtonDoubleClickEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::MiddleButtonDoubleClickEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::RightButtonDoubleClickEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::MouseWheelForwardEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::MouseWheelBackwardEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::MouseWheelLeftEvent, this->mouseObserver);
    this->interactor->AddObserver(vtkCommand::MouseWheelRightEvent, this->mouseObserver);

    this->exitObserver->SetClientData(this);
    this->exitObserver->SetCallback(&Impl::HandleExitEvent);
    this->interactor->AddObserver(vtkCommand::ExitEvent, this->exitObserver);
  }

  void ensureInitialized() {
    if (this->initialized) {
      return;
    }
    this->renderWindow->Render();
    this->interactor->Initialize();
    this->initialized = true;
  }

  static void HandleKeyEvent(vtkObject* caller, unsigned long eventId, void* clientData, void*) {
    static_cast<Impl*>(clientData)->dispatchKeyEvent(caller, eventId);
  }

  static void HandleMouseEvent(vtkObject* caller, unsigned long eventId, void* clientData, void*) {
    static_cast<Impl*>(clientData)->dispatchMouseEvent(caller, eventId);
  }

  static void HandleExitEvent(vtkObject*, unsigned long, void* clientData, void*) {
    static_cast<Impl*>(clientData)->stopped = true;
  }

  void dispatchKeyEvent(vtkObject* caller, unsigned long eventId) {
    auto* vtkInteractor = vtkRenderWindowInteractor::SafeDownCast(caller);
    if (vtkInteractor == nullptr) {
      return;
    }

    KeyEvent event;
    event.action = eventId == vtkCommand::KeyReleaseEvent ? KeyAction::Release : KeyAction::Press;
    event.modifiers = makeModifiers(vtkInteractor);
    event.repeatCount = vtkInteractor->GetRepeatCount();
    event.keyCode = vtkInteractor->GetKeyCode();
    const char* keySym = vtkInteractor->GetKeySym();
    if (keySym != nullptr) {
      event.keySym = keySym;
    }

    std::vector<KeyEventCallback> callbacks;
    callbacks.reserve(this->keyCallbacks.size());
    for (const auto& [_, callback] : this->keyCallbacks) {
      callbacks.push_back(callback);
    }

    for (const auto& callback : callbacks) {
      if (callback) {
        callback(event);
      }
    }

    this->handleDefaultHotkeys(event);
  }

  void dispatchMouseEvent(vtkObject* caller, unsigned long eventId) {
    auto* vtkInteractor = vtkRenderWindowInteractor::SafeDownCast(caller);
    if (vtkInteractor == nullptr || this->mouseCallbacks.empty()) {
      return;
    }

    const int* eventPosition = vtkInteractor->GetEventPosition();
    const int* lastEventPosition = vtkInteractor->GetLastEventPosition();

    MouseEvent event;
    event.action = makeMouseAction(eventId);
    event.button = makeMouseButton(eventId);
    event.modifiers = makeModifiers(vtkInteractor);
    event.x = eventPosition[0];
    event.y = eventPosition[1];
    event.lastX = lastEventPosition[0];
    event.lastY = lastEventPosition[1];
    event.wheelDelta = makeWheelDelta(eventId);
    event.repeatCount = vtkInteractor->GetRepeatCount();

    std::vector<MouseEventCallback> callbacks;
    callbacks.reserve(this->mouseCallbacks.size());
    for (const auto& [_, callback] : this->mouseCallbacks) {
      callbacks.push_back(callback);
    }

    for (const auto& callback : callbacks) {
      if (callback) {
        callback(event);
      }
    }
  }

  void requestClose() {
    this->stopped = true;
    this->interactor->TerminateApp();
  }

  void handleDefaultHotkeys(const KeyEvent& event) {
    if (!this->defaultHotkeysEnabled || event.action != KeyAction::Press) {
      return;
    }

    if (event.keySym == "Escape") {
      this->requestClose();
      return;
    }

    if (event.keySym == "a" || event.keySym == "A") {
      this->axesEnabled = !this->axesEnabled;
      this->axesWidget->SetEnabled(this->axesEnabled ? 1 : 0);
      this->renderWindow->Render();
      return;
    }

    if (event.keySym == "plus" || event.keySym == "equal" || event.keySym == "KP_Add") {
      this->applySizeScaleFactor(1.1f);
      return;
    }

    if (event.keySym == "minus" || event.keySym == "underscore" || event.keySym == "KP_Subtract") {
      this->applySizeScaleFactor(1.0f / 1.1f);
    }
  }

  void applySizeScaleFactor(float factor) {
    if (!std::isfinite(factor) || factor <= 0.0f) {
      return;
    }

    for (auto& [_, cloud] : this->clouds) {
      cloud.options.sizeScale *= factor;
      cloud.prop->SetRenderOptions(cloud.options);
    }

    if (!this->clouds.empty()) {
      this->renderWindow->Render();
    }
  }

  std::string windowName;
  vtkSmartPointer<vtkRenderer> renderer;
  vtkSmartPointer<vtkRenderWindow> renderWindow;
  vtkSmartPointer<vtkRenderWindowInteractor> interactor;
  vtkSmartPointer<vtkInteractorStyleTrackballCamera> interactorStyle;
  vtkSmartPointer<vtkAxesActor> axesActor;
  vtkSmartPointer<vtkOrientationMarkerWidget> axesWidget;
  vtkSmartPointer<vtkCallbackCommand> keyObserver;
  vtkSmartPointer<vtkCallbackCommand> mouseObserver;
  vtkSmartPointer<vtkCallbackCommand> exitObserver;
  std::unordered_map<std::string, CloudEntry> clouds;
  std::unordered_map<KeyCallbackHandle, KeyEventCallback> keyCallbacks;
  std::unordered_map<MouseCallbackHandle, MouseEventCallback> mouseCallbacks;
  CallbackHandle nextCallbackHandle{1};
  std::array<double, 3> backgroundColor{0.07, 0.08, 0.10};
  double axesLength{1.0};
  bool axesEnabled{true};
  bool defaultHotkeysEnabled{true};
  bool initialized{false};
  bool stopped{false};
};

SplatVisualizer::SplatVisualizer(std::string windowName) : impl_(std::make_unique<Impl>(std::move(windowName))) {}

SplatVisualizer::~SplatVisualizer() = default;

SplatVisualizer::SplatVisualizer(SplatVisualizer&&) noexcept = default;

SplatVisualizer& SplatVisualizer::operator=(SplatVisualizer&&) noexcept = default;

bool SplatVisualizer::addSplatCloud(std::shared_ptr<const DataTable> dataTable, const std::string& id,
                                    const SplatRenderOptions& options) {
  if (dataTable == nullptr) {
    throw std::invalid_argument("dataTable must not be null.");
  }
  if (id.empty()) {
    throw std::invalid_argument("Cloud id must not be empty.");
  }
  if (this->impl_->clouds.find(id) != this->impl_->clouds.end()) {
    return false;
  }

  auto prop = vtkSmartPointer<NativeSplatProp>::New();
  prop->SetInputData(dataTable);
  prop->SetRenderOptions(options);

  this->impl_->renderer->AddViewProp(prop);
  this->impl_->clouds.emplace(id, CloudEntry{std::move(dataTable), prop, options});
  return true;
}

bool SplatVisualizer::addSplatCloud(const DataTable& dataTable, const std::string& id,
                                    const SplatRenderOptions& options) {
  return this->addSplatCloud(cloneShared(dataTable), id, options);
}

bool SplatVisualizer::updateSplatCloud(std::shared_ptr<const DataTable> dataTable, const std::string& id,
                                       const SplatRenderOptions& options) {
  if (dataTable == nullptr) {
    throw std::invalid_argument("dataTable must not be null.");
  }

  auto it = this->impl_->clouds.find(id);
  if (it == this->impl_->clouds.end()) {
    return false;
  }

  it->second.dataTable = std::move(dataTable);
  it->second.options = options;
  it->second.prop->SetInputData(it->second.dataTable);
  it->second.prop->SetRenderOptions(it->second.options);
  return true;
}

bool SplatVisualizer::updateSplatCloud(const DataTable& dataTable, const std::string& id,
                                       const SplatRenderOptions& options) {
  return this->updateSplatCloud(cloneShared(dataTable), id, options);
}

bool SplatVisualizer::contains(const std::string& id) const {
  return this->impl_->clouds.find(id) != this->impl_->clouds.end();
}

bool SplatVisualizer::removeSplatCloud(const std::string& id) {
  const auto it = this->impl_->clouds.find(id);
  if (it == this->impl_->clouds.end()) {
    return false;
  }

  this->impl_->renderer->RemoveViewProp(it->second.prop);
  this->impl_->clouds.erase(it);
  return true;
}

void SplatVisualizer::removeAllSplatClouds() {
  for (const auto& [_, entry] : this->impl_->clouds) {
    this->impl_->renderer->RemoveViewProp(entry.prop);
  }
  this->impl_->clouds.clear();
}

std::vector<std::string> SplatVisualizer::getSplatCloudIds() const {
  std::vector<std::string> ids;
  ids.reserve(this->impl_->clouds.size());
  for (const auto& [id, _] : this->impl_->clouds) {
    ids.push_back(id);
  }
  return ids;
}

size_t SplatVisualizer::getSplatCount(const std::string& id) const {
  const auto it = this->impl_->clouds.find(id);
  if (it == this->impl_->clouds.end()) {
    return 0;
  }
  return it->second.prop->GetSplatCount();
}

bool SplatVisualizer::setSplatRenderOptions(const std::string& id, const SplatRenderOptions& options) {
  auto it = this->impl_->clouds.find(id);
  if (it == this->impl_->clouds.end()) {
    return false;
  }

  it->second.options = options;
  it->second.prop->SetRenderOptions(it->second.options);
  return true;
}

void SplatVisualizer::multiplySplatSizeScale(float factor) { this->impl_->applySizeScaleFactor(factor); }

void SplatVisualizer::setBackgroundColor(double r, double g, double b) {
  this->impl_->backgroundColor = {r, g, b};
  this->impl_->renderer->SetBackground(r, g, b);
}

void SplatVisualizer::getBackgroundColor(double& r, double& g, double& b) const {
  r = this->impl_->backgroundColor[0];
  g = this->impl_->backgroundColor[1];
  b = this->impl_->backgroundColor[2];
}

void SplatVisualizer::setWindowSize(int width, int height) { this->impl_->renderWindow->SetSize(width, height); }

void SplatVisualizer::setWindowName(const std::string& windowName) {
  this->impl_->windowName = windowName;
  this->impl_->renderWindow->SetWindowName(this->impl_->windowName.c_str());
}

const std::string& SplatVisualizer::getWindowName() const { return this->impl_->windowName; }

void SplatVisualizer::setAxesEnabled(bool enabled) {
  this->impl_->axesEnabled = enabled;
  this->impl_->axesWidget->SetEnabled(enabled ? 1 : 0);
}

void SplatVisualizer::setAxesLength(double length) {
  this->impl_->axesLength = length;
  this->impl_->axesActor->SetTotalLength(length, length, length);
}

void SplatVisualizer::setDefaultHotkeysEnabled(bool enabled) { this->impl_->defaultHotkeysEnabled = enabled; }

bool SplatVisualizer::getDefaultHotkeysEnabled() const { return this->impl_->defaultHotkeysEnabled; }

void SplatVisualizer::resetCamera() { this->impl_->renderer->ResetCamera(); }

void SplatVisualizer::render() {
  this->impl_->ensureInitialized();
  this->impl_->renderWindow->Render();
}

void SplatVisualizer::spin() {
  this->impl_->ensureInitialized();
  this->impl_->renderWindow->Render();
  this->impl_->stopped = false;
  this->impl_->interactor->Start();
  this->impl_->stopped = this->impl_->stopped || this->impl_->interactor->GetDone();
}

void SplatVisualizer::spinOnce(int time, bool forceRedraw) {
  this->impl_->ensureInitialized();
  if (forceRedraw || this->impl_->renderWindow->GetNeverRendered()) {
    this->impl_->renderWindow->Render();
  }
  this->impl_->interactor->ProcessEvents();
  if (time > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(time));
  }
  if (forceRedraw) {
    this->impl_->renderWindow->Render();
  }
  this->impl_->stopped = this->impl_->stopped || this->impl_->interactor->GetDone();
}

void SplatVisualizer::close() {
  this->impl_->requestClose();
  this->impl_->renderWindow->Finalize();
}

bool SplatVisualizer::wasStopped() const { return this->impl_->stopped || this->impl_->interactor->GetDone(); }

SplatVisualizer::KeyCallbackHandle SplatVisualizer::registerKeyCallback(KeyEventCallback callback) {
  const auto handle = this->impl_->nextCallbackHandle++;
  this->impl_->keyCallbacks.emplace(handle, std::move(callback));
  return handle;
}

bool SplatVisualizer::unregisterKeyCallback(KeyCallbackHandle handle) {
  return this->impl_->keyCallbacks.erase(handle) > 0;
}

void SplatVisualizer::clearKeyCallbacks() { this->impl_->keyCallbacks.clear(); }

SplatVisualizer::MouseCallbackHandle SplatVisualizer::registerMouseCallback(MouseEventCallback callback) {
  const auto handle = this->impl_->nextCallbackHandle++;
  this->impl_->mouseCallbacks.emplace(handle, std::move(callback));
  return handle;
}

bool SplatVisualizer::unregisterMouseCallback(MouseCallbackHandle handle) {
  return this->impl_->mouseCallbacks.erase(handle) > 0;
}

void SplatVisualizer::clearMouseCallbacks() { this->impl_->mouseCallbacks.clear(); }

vtkRenderer* SplatVisualizer::getRenderer() const { return this->impl_->renderer; }

vtkRenderWindow* SplatVisualizer::getRenderWindow() const { return this->impl_->renderWindow; }

vtkRenderWindowInteractor* SplatVisualizer::getInteractor() const { return this->impl_->interactor; }

vtkOrientationMarkerWidget* SplatVisualizer::getAxesWidget() const { return this->impl_->axesWidget; }

}  // namespace splat
