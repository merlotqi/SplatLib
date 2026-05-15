#pragma once

#include <splat/visualization/gsplat_data.h>

#include <Eigen/Core>
#include <vtk_glew.h>

#include <cstdint>
#include <vector>

namespace splat::visualization {

struct GSplatGLRenderOptions {
  float globalOpacity = 1.0f;
  float sizeScale = 1.0f;
  float minPixelSize = 2.0f;
  float maxPixelSize = 1024.0f;
  float alphaDiscardThreshold = 1.0f / 255.0f;
  bool sortBackToFront = true;
  bool depthTest = true;
  bool depthWrite = false;
  bool clampColors = true;
};

struct GSplatGLFrameState {
  Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
  Eigen::Vector3f cameraPosition{0.0f, 0.0f, 0.0f};
  Eigen::Vector3f cameraForward{0.0f, 0.0f, -1.0f};
  int width = 1;
  int height = 1;
  float nearPlane = 0.001f;
  float farPlane = 1000.0f;
};

class GSplatGLRenderer {
 public:
  GSplatGLRenderer() = default;
  ~GSplatGLRenderer();

  GSplatGLRenderer(const GSplatGLRenderer&) = delete;
  GSplatGLRenderer& operator=(const GSplatGLRenderer&) = delete;

  void setData(const GSplatData& data);
  void render(const GSplatGLFrameState& frame, const GSplatGLRenderOptions& options);
  void releaseGraphicsResources();

 private:
  struct InstanceData {
    Vec3f center;
    Vec4f rotation;
    Vec3f scale;
    Vec4f color;
  };

  void ensureProgram();
  void ensureBuffers();
  bool needsInstanceOrderUpdate(const GSplatGLFrameState& frame, bool sortBackToFront) const;
  void updateSortedInstances(const GSplatGLFrameState& frame, bool sortBackToFront);
  void uploadInstances();

  GSplatData data_;
  std::vector<std::uint32_t> order_;
  std::vector<InstanceData> sortedInstances_;
  GLuint vao_ = 0;
  GLuint cornerBuffer_ = 0;
  GLuint instanceBuffer_ = 0;
  GLuint program_ = 0;
  bool instancesDirty_ = true;
  bool orderInitialized_ = false;
  bool lastSortBackToFront_ = true;
  Eigen::Vector3f lastSortPosition_{0.0f, 0.0f, 0.0f};
  Eigen::Vector3f lastSortForward_{0.0f, 0.0f, -1.0f};
};

}  // namespace splat::visualization