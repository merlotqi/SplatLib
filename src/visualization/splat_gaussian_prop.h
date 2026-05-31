#pragma once

#include <splat/visualization/splat_visualizer.h>

#include <cstddef>
#include <memory>

#include <vtkProp3D.h>

namespace splat {

class DataTable;

class SplatGaussianProp : public vtkProp3D {
 public:
  static SplatGaussianProp* New();
  vtkTypeMacro(SplatGaussianProp, vtkProp3D);

  void SetInputData(std::shared_ptr<const DataTable> dataTable);
  void SetInputData(const DataTable& dataTable);
  void SetRenderOptions(const SplatRenderOptions& options);
  const SplatRenderOptions& GetRenderOptions() const noexcept;
  std::size_t GetSplatCount() const noexcept;

  double* GetBounds() override;
  int RenderOpaqueGeometry(vtkViewport* viewport) override;
  int RenderTranslucentPolygonalGeometry(vtkViewport* viewport) override;
  vtkTypeBool HasTranslucentPolygonalGeometry() override;
  void ReleaseGraphicsResources(vtkWindow* window) override;

 protected:
  SplatGaussianProp();
  ~SplatGaussianProp() override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace splat
