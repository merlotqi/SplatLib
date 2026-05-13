#pragma once

#include <splat/visualization/keyevent.h>
#include <splat/visualization/mouseevent.h>

#include <memory>
#include <string>
#include <vector>

class vtkOrientationMarkerWidget;
class vtkRenderWindow;
class vtkRenderWindowInteractor;
class vtkRenderer;

namespace splat {

class DataTable;

struct SplatRenderOptions {
  float globalOpacity{1.0f};
  float sizeScale{3.0f};
  float minPointSize{1.0f};
  float maxPointSize{1024.0f};
  float alphaDiscardThreshold{0.001f};
  bool visible{true};
  bool depthTest{true};
  bool depthWrite{false};
};

class SplatVisualizer {
 public:
  using Ptr = std::shared_ptr<SplatVisualizer>;
  using ConstPtr = std::shared_ptr<const SplatVisualizer>;
  using CallbackHandle = std::size_t;
  using KeyCallbackHandle = std::size_t;
  using MouseCallbackHandle = std::size_t;

  explicit SplatVisualizer(std::string windowName = "SplatVisualizer");
  ~SplatVisualizer();

  SplatVisualizer(const SplatVisualizer&) = delete;
  SplatVisualizer& operator=(const SplatVisualizer&) = delete;
  SplatVisualizer(SplatVisualizer&&) noexcept;
  SplatVisualizer& operator=(SplatVisualizer&&) noexcept;

  bool addSplatCloud(std::shared_ptr<const DataTable> dataTable, const std::string& id = "splat",
                     const SplatRenderOptions& options = {});
  bool addSplatCloud(const DataTable& dataTable, const std::string& id = "splat",
                     const SplatRenderOptions& options = {});
  bool updateSplatCloud(std::shared_ptr<const DataTable> dataTable, const std::string& id,
                        const SplatRenderOptions& options = {});
  bool updateSplatCloud(const DataTable& dataTable, const std::string& id, const SplatRenderOptions& options = {});
  bool contains(const std::string& id) const;
  bool removeSplatCloud(const std::string& id);
  void removeAllSplatClouds();
  std::vector<std::string> getSplatCloudIds() const;
  size_t getSplatCount(const std::string& id) const;

  bool setSplatRenderOptions(const std::string& id, const SplatRenderOptions& options);
  void multiplySplatSizeScale(float factor);

  void setBackgroundColor(double r, double g, double b);
  void getBackgroundColor(double& r, double& g, double& b) const;
  void setWindowSize(int width, int height);
  void setWindowName(const std::string& windowName);
  const std::string& getWindowName() const;

  void setAxesEnabled(bool enabled);
  void setAxesLength(double length);
  void setDefaultHotkeysEnabled(bool enabled);
  bool getDefaultHotkeysEnabled() const;

  void resetCamera();
  void render();
  void spin();
  void spinOnce(int time = 1, bool forceRedraw = false);
  void close();
  bool wasStopped() const;

  KeyCallbackHandle registerKeyCallback(KeyEventCallback callback);
  bool unregisterKeyCallback(KeyCallbackHandle handle);
  void clearKeyCallbacks();
  MouseCallbackHandle registerMouseCallback(MouseEventCallback callback);
  bool unregisterMouseCallback(MouseCallbackHandle handle);
  void clearMouseCallbacks();

  vtkRenderer* getRenderer() const;
  vtkRenderWindow* getRenderWindow() const;
  vtkRenderWindowInteractor* getInteractor() const;
  vtkOrientationMarkerWidget* getAxesWidget() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace splat
