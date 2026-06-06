#include "splat_gaussian_prop.h"

#include <splat/models/splatcloud.h>
#include <splat/spatial/gaussian_aabb.h>
#include <vtkCamera.h>
#include <vtkMatrix4x4.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkOpenGLRenderWindow.h>
#include <vtkOpenGLState.h>
#include <vtkRenderer.h>
#include <vtkViewport.h>
#include <vtkWindow.h>

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "gsplat_data.h"
#include "gsplat_gl_renderer.h"

namespace splat {
namespace {

std::shared_ptr<const SplatCloud> cloneShared(const SplatCloud& dataTable) {
  auto clone = dataTable.clone();
  return std::shared_ptr<const SplatCloud>(clone.release());
}

bool isValidBounds(const std::array<double, 6>& bounds) {
  return std::isfinite(bounds[0]) && std::isfinite(bounds[1]) && std::isfinite(bounds[2]) && std::isfinite(bounds[3]) &&
         std::isfinite(bounds[4]) && std::isfinite(bounds[5]) && bounds[0] <= bounds[1] && bounds[2] <= bounds[3] &&
         bounds[4] <= bounds[5];
}

std::array<double, 6> computeBoundsFromGSplatData(const visualization::GSplatData& data) {
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

std::array<double, 6> computeSplatBounds(const SplatCloud& dataTable, const visualization::GSplatData& data) {
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

void transformPoint(const double matrix[16], const std::array<double, 3>& point, double out[3]) {
  out[0] = matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2] + matrix[3];
  out[1] = matrix[4] * point[0] + matrix[5] * point[1] + matrix[6] * point[2] + matrix[7];
  out[2] = matrix[8] * point[0] + matrix[9] * point[1] + matrix[10] * point[2] + matrix[11];
}

}  // namespace

class SplatGaussianProp::Impl {
 public:
  void SetInputData(std::shared_ptr<const SplatCloud> dataTable) {
    this->inputDataTable = std::move(dataTable);
    this->RebuildCpuCache();
  }

  void SetRenderOptions(const SplatRenderOptions& newOptions) { this->options = newOptions; }

  double* GetBounds(vtkProp3D* owner) {
    if (!this->boundsValid) {
      return nullptr;
    }

    const double localBounds[6] = {
        this->localBounds[0], this->localBounds[1], this->localBounds[2],
        this->localBounds[3], this->localBounds[4], this->localBounds[5],
    };

    double modelMatrix[16];
    owner->GetMatrix(modelMatrix);

    const std::array<double, 8 * 3> corners = {
        localBounds[0], localBounds[2], localBounds[4], localBounds[1], localBounds[2], localBounds[4],
        localBounds[0], localBounds[3], localBounds[4], localBounds[1], localBounds[3], localBounds[4],
        localBounds[0], localBounds[2], localBounds[5], localBounds[1], localBounds[2], localBounds[5],
        localBounds[0], localBounds[3], localBounds[5], localBounds[1], localBounds[3], localBounds[5],
    };

    this->worldBounds = {
        std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
        std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(),
    };

    for (std::size_t i = 0; i < corners.size(); i += 3) {
      double transformed[3];
      transformPoint(modelMatrix, {corners[i + 0], corners[i + 1], corners[i + 2]}, transformed);
      this->worldBounds[0] = std::min(this->worldBounds[0], transformed[0]);
      this->worldBounds[1] = std::max(this->worldBounds[1], transformed[0]);
      this->worldBounds[2] = std::min(this->worldBounds[2], transformed[1]);
      this->worldBounds[3] = std::max(this->worldBounds[3], transformed[1]);
      this->worldBounds[4] = std::min(this->worldBounds[4], transformed[2]);
      this->worldBounds[5] = std::max(this->worldBounds[5], transformed[2]);
    }

    return this->worldBounds.data();
  }

  int RenderSplatGeometry(vtkProp3D* owner, vtkViewport* viewport) {
    if (!owner->GetVisibility() || this->renderData.empty()) {
      return 0;
    }

    auto* renderer = vtkRenderer::SafeDownCast(viewport);
    if (renderer == nullptr) {
      return 0;
    }

    auto* openGLWindow = vtkOpenGLRenderWindow::SafeDownCast(renderer->GetRenderWindow());
    if (openGLWindow == nullptr) {
      throw std::runtime_error("SplatGaussianProp requires a vtkOpenGLRenderWindow.");
    }

    openGLWindow->MakeCurrent();
    auto* state = openGLWindow->GetState();
    state->Push();

    int viewportWidth = 1;
    int viewportHeight = 1;
    int viewportLowerLeftX = 0;
    int viewportLowerLeftY = 0;
    renderer->GetTiledSizeAndOrigin(&viewportWidth, &viewportHeight, &viewportLowerLeftX, &viewportLowerLeftY);
    const double aspect =
        static_cast<double>(std::max(viewportWidth, 1)) / static_cast<double>(std::max(viewportHeight, 1));

    auto* camera = renderer->GetActiveCamera();
    double clippingRange[2] = {0.001, 1000.0};
    camera->GetClippingRange(clippingRange);
    auto* viewToClipMatrix = camera->GetProjectionTransformMatrix(aspect, clippingRange[0], clippingRange[1]);

    vtkNew<vtkMatrix4x4> modelMatrix;
    vtkNew<vtkMatrix4x4> modelViewMatrix;
    owner->GetMatrix(modelMatrix);
    vtkMatrix4x4::Multiply4x4(camera->GetViewTransformMatrix(), modelMatrix, modelViewMatrix);
    double cameraPosition[3] = {0.0, 0.0, 0.0};
    double cameraForward[3] = {0.0, 0.0, -1.0};
    camera->GetPosition(cameraPosition);
    camera->GetDirectionOfProjection(cameraForward);

    visualization::GSplatGLFrameState frame;
    frame.view = copyMatrixToEigen(modelViewMatrix);
    frame.projection = copyMatrixToEigen(viewToClipMatrix);
    frame.cameraPosition = Eigen::Vector3f(static_cast<float>(cameraPosition[0]), static_cast<float>(cameraPosition[1]),
                                           static_cast<float>(cameraPosition[2]));
    frame.cameraForward = Eigen::Vector3f(static_cast<float>(cameraForward[0]), static_cast<float>(cameraForward[1]),
                                          static_cast<float>(cameraForward[2]));
    frame.width = std::max(viewportWidth, 1);
    frame.height = std::max(viewportHeight, 1);
    frame.nearPlane = static_cast<float>(clippingRange[0]);
    frame.farPlane = static_cast<float>(clippingRange[1]);

    visualization::GSplatGLRenderOptions renderOptions;
    renderOptions.globalOpacity = this->options.globalOpacity;
    renderOptions.sizeScale = this->options.sizeScale;
    renderOptions.minPixelSize = this->options.minPointSize;
    renderOptions.maxPixelSize = this->options.maxPointSize;
    renderOptions.alphaDiscardThreshold = this->options.alphaDiscardThreshold;
    renderOptions.sortBackToFront = this->options.sortBackToFront;
    renderOptions.depthTest = this->options.depthTest;
    renderOptions.depthWrite = this->options.depthWrite;
    renderOptions.clampColors = this->options.clampColors;
    renderOptions.freezeSortOrder = this->options.freezeSortOrder;

    this->renderer.render(frame, renderOptions);
    state->Pop();
    return 1;
  }

  void ReleaseGraphicsResources(vtkWindow* window) {
    auto* openGLWindow = vtkOpenGLRenderWindow::SafeDownCast(window);
    if (openGLWindow == nullptr) {
      return;
    }

    openGLWindow->MakeCurrent();
    this->renderer.releaseGraphicsResources();
  }

  void RebuildCpuCache() {
    this->renderData = {};
    this->renderer.setData(this->renderData);
    this->boundsValid = false;
    this->localBounds = {1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
    this->worldBounds = this->localBounds;

    if (!this->inputDataTable || this->inputDataTable->getNumRows() == 0) {
      return;
    }

    try {
      this->renderData = visualization::adaptDataTableToGSplatData(*this->inputDataTable);
    } catch (const std::runtime_error& error) {
      std::string message = error.what();
      const std::string prefix = "GSplatData: ";
      if (message.rfind(prefix, 0) == 0) {
        message.replace(0, prefix.size(), "SplatGaussianProp: ");
      }
      throw std::runtime_error(message);
    }

    if (this->renderData.empty()) {
      throw std::runtime_error("SplatGaussianProp: upload produced zero splats");
    }

    this->renderer.setData(this->renderData);
    this->localBounds = computeSplatBounds(*this->inputDataTable, this->renderData);
    this->boundsValid = isValidBounds(this->localBounds);
  }

  std::shared_ptr<const SplatCloud> inputDataTable;
  SplatRenderOptions options;
  visualization::GSplatData renderData;
  visualization::GSplatGLRenderer renderer;
  bool boundsValid{false};
  std::array<double, 6> localBounds{1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
  std::array<double, 6> worldBounds{1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
};

vtkStandardNewMacro(SplatGaussianProp);

SplatGaussianProp::SplatGaussianProp() : impl_(std::make_unique<Impl>()) {}

SplatGaussianProp::~SplatGaussianProp() = default;

void SplatGaussianProp::SetInputData(std::shared_ptr<const SplatCloud> dataTable) {
  this->impl_->SetInputData(std::move(dataTable));
  this->Modified();
}

void SplatGaussianProp::SetInputData(const SplatCloud& dataTable) { this->SetInputData(cloneShared(dataTable)); }

void SplatGaussianProp::SetRenderOptions(const SplatRenderOptions& options) {
  this->impl_->SetRenderOptions(options);
  this->SetVisibility(options.visible ? 1 : 0);
  this->Modified();
}

const SplatRenderOptions& SplatGaussianProp::GetRenderOptions() const noexcept { return this->impl_->options; }

std::size_t SplatGaussianProp::GetSplatCount() const noexcept { return this->impl_->renderData.size(); }

double* SplatGaussianProp::GetBounds() { return this->impl_->GetBounds(this); }

int SplatGaussianProp::RenderOpaqueGeometry(vtkViewport* viewport) {
  return this->impl_->RenderSplatGeometry(this, viewport);
}

int SplatGaussianProp::RenderTranslucentPolygonalGeometry(vtkViewport*) { return 0; }

vtkTypeBool SplatGaussianProp::HasTranslucentPolygonalGeometry() { return 0; }

void SplatGaussianProp::ReleaseGraphicsResources(vtkWindow* window) { this->impl_->ReleaseGraphicsResources(window); }

}  // namespace splat
