#include <splat/models/data-table.h>
#include <splat/spatial/gaussian_aabb.h>
#include <splat/visualization/gsplat_data.h>
#include <splat/visualization/gsplat_gl_renderer.h>
#include <splat/visualization/splat_gaussian_prop.h>
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
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace splat {
namespace {

constexpr double kPlayCanvasFovYDegrees = 60.0;
constexpr double kPi = 3.141592653589793238462643383279502884;

std::shared_ptr<const DataTable> cloneShared(const DataTable& dataTable) {
  auto clone = dataTable.clone();
  return std::shared_ptr<const DataTable>(clone.release());
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

struct CloudEntry {
  std::shared_ptr<const DataTable> dataTable;
  vtkSmartPointer<SplatGaussianProp> prop;
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
    this->initializeVtkObjects();
  }

  Impl(vtkRenderer* externalRenderer, vtkRenderWindow* externalRenderWindow,
       vtkRenderWindowInteractor* externalInteractor, std::string name)
      : windowName(std::move(name)),
        renderer(externalRenderer),
        renderWindow(externalRenderWindow),
        interactor(externalInteractor),
        interactorStyle(vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New()),
        axesActor(vtkSmartPointer<vtkAxesActor>::New()),
        axesWidget(vtkSmartPointer<vtkOrientationMarkerWidget>::New()),
        keyObserver(vtkSmartPointer<vtkCallbackCommand>::New()),
        mouseObserver(vtkSmartPointer<vtkCallbackCommand>::New()),
        exitObserver(vtkSmartPointer<vtkCallbackCommand>::New()) {
    if (this->renderer == nullptr || this->renderWindow == nullptr || this->interactor == nullptr) {
      throw std::invalid_argument("SplatVisualizer external VTK objects must not be null.");
    }
    this->initializeVtkObjects();
  }

  void initializeVtkObjects() {
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

SplatVisualizer::SplatVisualizer(vtkRenderer* renderer, vtkRenderWindow* renderWindow,
                 vtkRenderWindowInteractor* interactor, std::string windowName)
  : impl_(std::make_unique<Impl>(renderer, renderWindow, interactor, std::move(windowName))) {}

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

  auto prop = vtkSmartPointer<SplatGaussianProp>::New();
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

void SplatVisualizer::resetCameraToBounds(const double bounds[6]) {
  if (bounds == nullptr) {
    this->resetCamera();
    return;
  }

  const double dx = bounds[1] - bounds[0];
  const double dy = bounds[3] - bounds[2];
  const double dz = bounds[5] - bounds[4];
  const double radius = std::max(0.01, 0.5 * std::sqrt(dx * dx + dy * dy + dz * dz));
  const double center[3] = {
      0.5 * (bounds[0] + bounds[1]),
      0.5 * (bounds[2] + bounds[3]),
      0.5 * (bounds[4] + bounds[5]),
  };

  const int* size = this->impl_->renderWindow->GetSize();
  const double aspect = static_cast<double>(std::max(size[0], 1)) / static_cast<double>(std::max(size[1], 1));
  const double verticalHalfFov = (kPlayCanvasFovYDegrees * kPi / 180.0) * 0.5;
  const double horizontalHalfFov = std::atan(std::tan(verticalHalfFov) * aspect);
  const double fitHalfFov = std::max(0.001, std::min(verticalHalfFov, horizontalHalfFov));
  const double distance = std::max((radius * 1.5) / std::sin(fitHalfFov), 1.0);

  auto* camera = this->impl_->renderer->GetActiveCamera();
  camera->SetViewAngle(kPlayCanvasFovYDegrees);
  camera->SetPosition(center[0], center[1], center[2] + distance);
  camera->SetFocalPoint(center[0], center[1], center[2]);
  camera->SetViewUp(0.0, -1.0, 0.0);
  camera->SetClippingRange(std::max(std::min(radius * 0.0005, 0.01), 0.0001),
                           std::max({distance + radius * 4.0, radius * 32.0, 10.0}));
}

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
