# PlayCanvas Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent `PlaycanvasViewer` executable that renders visible 3D Gaussian splats by translating the PlayCanvas GSplat data and shader semantics into a focused C++/OpenGL path.

**Architecture:** Add a new top-level `playcanvasViewer/` tool that links `SPLAT::splat` and owns its own DataTable adapter, CPU sort, camera controller, OpenGL shader sources, renderer, and GLFW main loop. The first version uses OpenGL 3.3 instanced attributes with CPU reordering after sort, preserving PlayCanvas decode and covariance projection semantics without depending on VTK or WebGPU-style texture indirection.

**Tech Stack:** C++17, CMake, SplatLib public readers, `splat::DataTable`, Eigen, GLFW, GLEW, OpenGL 3.3, CTest

---

## File Structure

**Create:**
- `playcanvasViewer/CMakeLists.txt` - build `playcanvas_viewer_adapter` and `PlaycanvasViewer`.
- `playcanvasViewer/gsplat_data_adapter.h` - testable DataTable-to-render-data API.
- `playcanvasViewer/gsplat_data_adapter.cpp` - PlayCanvas decode semantics for position, rotation, scale, color, and opacity.
- `playcanvasViewer/gsplat_shader_sources.h` - GLSL strings translated from PlayCanvas center/corner/falloff chunks.
- `playcanvasViewer/gsplat_renderer.h` - OpenGL renderer API and render options.
- `playcanvasViewer/gsplat_renderer.cpp` - shader compilation, GPU uploads, CPU sorted instance buffers, draw submission.
- `playcanvasViewer/camera_controller.h` - camera state/control API.
- `playcanvasViewer/camera_controller.cpp` - orbit, dolly, fly, bounds reset, view/projection matrices.
- `playcanvasViewer/main.cpp` - CLI, file reader dispatch, GLFW/GLEW setup, callbacks, render loop.
- `tests/playcanvas_viewer_adapter_test.cpp` - adapter semantic regression tests.

**Modify:**
- `CMakeLists.txt` - add `BUILD_SPLAT_PLAYCANVAS_VIEWER` and subdirectory wiring.
- `tests/CMakeLists.txt` - add adapter tests when the adapter target exists.

**Do not modify:**
- `include/splat/visualization/*`
- `src/visualization/*`
- `viewer/*`

---

### Task 1: Add The Testable GSplat Adapter Target

**Files:**
- Create: `playcanvasViewer/CMakeLists.txt`
- Create: `playcanvasViewer/gsplat_data_adapter.h`
- Create: `playcanvasViewer/gsplat_data_adapter.cpp`
- Modify: `CMakeLists.txt`
- Test: CMake configure/build target existence

- [ ] **Step 1: Add the top-level build option**

In `CMakeLists.txt`, add the option beside the existing tool options:

```cmake
option(BUILD_SPLAT_PLAYCANVAS_VIEWER "Build PlayCanvas-style OpenGL splat viewer" OFF)
```

Then add the subdirectory block after `playcanvasLOD` and before tests:

```cmake
if(BUILD_SPLAT_PLAYCANVAS_VIEWER)
    add_subdirectory(playcanvasViewer)
endif()
```

- [ ] **Step 2: Create `playcanvasViewer/CMakeLists.txt` with the adapter library first**

```cmake
cmake_minimum_required(VERSION 3.14)

add_library(playcanvas_viewer_adapter STATIC
    gsplat_data_adapter.cpp
)

target_include_directories(playcanvas_viewer_adapter
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(playcanvas_viewer_adapter
    PUBLIC
        SPLAT::splat
        Eigen3::Eigen
)
```

- [ ] **Step 3: Create `playcanvasViewer/gsplat_data_adapter.h`**

```cpp
#pragma once

#include <splat/models/data-table.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace playcanvas_viewer {

struct Vec3f {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct Vec4f {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 0.0f;
};

struct GSplatRenderData {
  std::vector<Vec3f> centers;
  std::vector<Vec4f> rotations;  // PlayCanvas source layout: w, x, y, z.
  std::vector<Vec3f> scales;
  std::vector<Vec4f> colors;     // Linear color and decoded alpha.
  std::vector<uint32_t> sourceIndices;

  size_t size() const { return centers.size(); }
  bool empty() const { return centers.empty(); }
};

GSplatRenderData adaptDataTableToGSplat(const splat::DataTable& dataTable, size_t maxSplats = 0);

float decodePlayCanvasColor(float dcValue);
float decodePlayCanvasOpacity(float opacityValue);
float decodePlayCanvasScale(float scaleValue);
Vec4f normalizePlayCanvasRotation(float w, float x, float y, float z);

}  // namespace playcanvas_viewer
```

- [ ] **Step 4: Create `playcanvasViewer/gsplat_data_adapter.cpp`**

```cpp
#include "gsplat_data_adapter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace playcanvas_viewer {
namespace {

constexpr float kSHC0 = 0.28209479177387814f;
constexpr const char* kRequiredColumns[] = {
    "x",       "y",       "z",       "rot_0",   "rot_1",  "rot_2", "rot_3",
    "scale_0", "scale_1", "scale_2", "f_dc_0",  "f_dc_1", "f_dc_2", "opacity"};

void requireColumns(const splat::DataTable& dataTable) {
  for (const char* name : kRequiredColumns) {
    if (!dataTable.hasColumn(name)) {
      throw std::runtime_error(std::string("PlaycanvasViewer: missing required column: ") + name);
    }
  }
}

float columnValue(const splat::DataTable& dataTable, const char* name, size_t row) {
  return dataTable.getColumnByName(name).getValue<float>(row);
}

}  // namespace

float decodePlayCanvasColor(float dcValue) { return 0.5f + dcValue * kSHC0; }

float decodePlayCanvasOpacity(float opacityValue) {
  if (opacityValue > 0.0f) {
    return 1.0f / (1.0f + std::exp(-opacityValue));
  }
  const float t = std::exp(opacityValue);
  return t / (1.0f + t);
}

float decodePlayCanvasScale(float scaleValue) { return std::exp(scaleValue); }

Vec4f normalizePlayCanvasRotation(float w, float x, float y, float z) {
  const float lengthSquared = w * w + x * x + y * y + z * z;
  if (!std::isfinite(lengthSquared) || lengthSquared <= std::numeric_limits<float>::epsilon()) {
    return {1.0f, 0.0f, 0.0f, 0.0f};
  }

  const float invLength = 1.0f / std::sqrt(lengthSquared);
  Vec4f rotation{w * invLength, x * invLength, y * invLength, z * invLength};
  if (rotation.x < 0.0f) {
    rotation.x = -rotation.x;
    rotation.y = -rotation.y;
    rotation.z = -rotation.z;
    rotation.w = -rotation.w;
  }
  return rotation;
}

GSplatRenderData adaptDataTableToGSplat(const splat::DataTable& dataTable, size_t maxSplats) {
  requireColumns(dataTable);

  const size_t rowCount = dataTable.getNumRows();
  const size_t count = maxSplats == 0 ? rowCount : std::min(rowCount, maxSplats);

  GSplatRenderData result;
  result.centers.reserve(count);
  result.rotations.reserve(count);
  result.scales.reserve(count);
  result.colors.reserve(count);
  result.sourceIndices.reserve(count);

  for (size_t row = 0; row < count; ++row) {
    result.centers.push_back({columnValue(dataTable, "x", row), columnValue(dataTable, "y", row),
                              columnValue(dataTable, "z", row)});

    result.rotations.push_back(normalizePlayCanvasRotation(columnValue(dataTable, "rot_0", row),
                                                           columnValue(dataTable, "rot_1", row),
                                                           columnValue(dataTable, "rot_2", row),
                                                           columnValue(dataTable, "rot_3", row)));

    result.scales.push_back({decodePlayCanvasScale(columnValue(dataTable, "scale_0", row)),
                             decodePlayCanvasScale(columnValue(dataTable, "scale_1", row)),
                             decodePlayCanvasScale(columnValue(dataTable, "scale_2", row))});

    result.colors.push_back({decodePlayCanvasColor(columnValue(dataTable, "f_dc_0", row)),
                             decodePlayCanvasColor(columnValue(dataTable, "f_dc_1", row)),
                             decodePlayCanvasColor(columnValue(dataTable, "f_dc_2", row)),
                             decodePlayCanvasOpacity(columnValue(dataTable, "opacity", row))});

    result.sourceIndices.push_back(static_cast<uint32_t>(row));
  }

  return result;
}

}  // namespace playcanvas_viewer
```

- [ ] **Step 5: Configure and build the adapter target**

Run:

```bash
cmake -S . -B build/playcanvas-viewer-plan -DBUILD_SPLAT_PLAYCANVAS_VIEWER=ON -DBUILD_SPLAT_TESTS=OFF
cmake --build build/playcanvas-viewer-plan --target playcanvas_viewer_adapter -j
```

Expected: `Built target playcanvas_viewer_adapter`.

- [ ] **Step 6: Commit the adapter scaffold**

```bash
git add CMakeLists.txt playcanvasViewer/CMakeLists.txt playcanvasViewer/gsplat_data_adapter.h playcanvasViewer/gsplat_data_adapter.cpp
git commit -m "feat: add PlayCanvas viewer data adapter"
```

### Task 2: Add Adapter Semantic Tests

**Files:**
- Create: `tests/playcanvas_viewer_adapter_test.cpp`
- Modify: `tests/CMakeLists.txt`
- Test: `PlaycanvasViewerAdapterTests`

- [ ] **Step 1: Create `tests/playcanvas_viewer_adapter_test.cpp`**

```cpp
#include "gsplat_data_adapter.h"

#include <splat/models/data-table.h>

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

bool near(float lhs, float rhs, float tolerance = 1e-5f) { return std::abs(lhs - rhs) <= tolerance; }

splat::DataTable makeOneRowTable() {
  splat::DataTable table;
  table.addColumn({"x", std::vector<float>{1.0f}});
  table.addColumn({"y", std::vector<float>{2.0f}});
  table.addColumn({"z", std::vector<float>{3.0f}});
  table.addColumn({"rot_0", std::vector<float>{-2.0f}});
  table.addColumn({"rot_1", std::vector<float>{0.0f}});
  table.addColumn({"rot_2", std::vector<float>{0.0f}});
  table.addColumn({"rot_3", std::vector<float>{0.0f}});
  table.addColumn({"scale_0", std::vector<float>{std::log(2.0f)}});
  table.addColumn({"scale_1", std::vector<float>{std::log(3.0f)}});
  table.addColumn({"scale_2", std::vector<float>{std::log(4.0f)}});
  table.addColumn({"f_dc_0", std::vector<float>{0.25f}});
  table.addColumn({"f_dc_1", std::vector<float>{0.0f}});
  table.addColumn({"f_dc_2", std::vector<float>{-0.25f}});
  table.addColumn({"opacity", std::vector<float>{0.0f}});
  return table;
}

void testDecodeHelpers() {
  using namespace playcanvas_viewer;
  assert(near(decodePlayCanvasColor(0.0f), 0.5f));
  assert(near(decodePlayCanvasOpacity(0.0f), 0.5f));
  assert(near(decodePlayCanvasScale(std::log(5.0f)), 5.0f));

  const auto rotation = normalizePlayCanvasRotation(-2.0f, 0.0f, 0.0f, 0.0f);
  assert(near(rotation.x, 1.0f));
  assert(near(rotation.y, 0.0f));
  assert(near(rotation.z, 0.0f));
  assert(near(rotation.w, 0.0f));
}

void testAdaptOneRow() {
  using namespace playcanvas_viewer;
  auto table = makeOneRowTable();
  const auto data = adaptDataTableToGSplat(table);

  assert(data.size() == 1);
  assert(near(data.centers[0].x, 1.0f));
  assert(near(data.centers[0].y, 2.0f));
  assert(near(data.centers[0].z, 3.0f));
  assert(near(data.scales[0].x, 2.0f));
  assert(near(data.scales[0].y, 3.0f));
  assert(near(data.scales[0].z, 4.0f));
  assert(near(data.colors[0].x, decodePlayCanvasColor(0.25f)));
  assert(near(data.colors[0].y, 0.5f));
  assert(near(data.colors[0].z, decodePlayCanvasColor(-0.25f)));
  assert(near(data.colors[0].w, 0.5f));
  assert(data.sourceIndices[0] == 0);
}

void testMaxSplatsLimit() {
  auto table = makeOneRowTable();
  table.getColumnByName("x").asVector<float>().push_back(8.0f);
  table.getColumnByName("y").asVector<float>().push_back(8.0f);
  table.getColumnByName("z").asVector<float>().push_back(8.0f);
  table.getColumnByName("rot_0").asVector<float>().push_back(1.0f);
  table.getColumnByName("rot_1").asVector<float>().push_back(0.0f);
  table.getColumnByName("rot_2").asVector<float>().push_back(0.0f);
  table.getColumnByName("rot_3").asVector<float>().push_back(0.0f);
  table.getColumnByName("scale_0").asVector<float>().push_back(0.0f);
  table.getColumnByName("scale_1").asVector<float>().push_back(0.0f);
  table.getColumnByName("scale_2").asVector<float>().push_back(0.0f);
  table.getColumnByName("f_dc_0").asVector<float>().push_back(0.0f);
  table.getColumnByName("f_dc_1").asVector<float>().push_back(0.0f);
  table.getColumnByName("f_dc_2").asVector<float>().push_back(0.0f);
  table.getColumnByName("opacity").asVector<float>().push_back(0.0f);

  const auto data = playcanvas_viewer::adaptDataTableToGSplat(table, 1);
  assert(data.size() == 1);
}

void testMissingColumnsFailClearly() {
  splat::DataTable table;
  table.addColumn({"x", std::vector<float>{1.0f}});

  try {
    (void)playcanvas_viewer::adaptDataTableToGSplat(table);
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    assert(message.find("missing required column") != std::string::npos);
    return;
  }

  assert(false && "Expected missing columns to throw");
}

}  // namespace

int main() {
  testDecodeHelpers();
  testAdaptOneRow();
  testMaxSplatsLimit();
  testMissingColumnsFailClearly();
  return 0;
}
```

- [ ] **Step 2: Wire the adapter test only when the target exists**

Append to `tests/CMakeLists.txt`:

```cmake
if(TARGET playcanvas_viewer_adapter)
    add_executable(PlaycanvasViewerAdapterTests playcanvas_viewer_adapter_test.cpp)
    target_link_libraries(PlaycanvasViewerAdapterTests PRIVATE playcanvas_viewer_adapter)
    add_test(NAME PlaycanvasViewerAdapterTests COMMAND PlaycanvasViewerAdapterTests)
endif()
```

- [ ] **Step 3: Build and run the adapter test**

Run:

```bash
cmake -S . -B build/playcanvas-viewer-plan -DBUILD_SPLAT_PLAYCANVAS_VIEWER=ON -DBUILD_SPLAT_TESTS=ON
cmake --build build/playcanvas-viewer-plan --target PlaycanvasViewerAdapterTests -j
ctest --test-dir build/playcanvas-viewer-plan --output-on-failure -R PlaycanvasViewerAdapterTests
```

Expected: `100% tests passed` for `PlaycanvasViewerAdapterTests`.

- [ ] **Step 4: Commit adapter tests**

```bash
git add tests/CMakeLists.txt tests/playcanvas_viewer_adapter_test.cpp
git commit -m "test: cover PlayCanvas viewer adapter semantics"
```

### Task 3: Add Camera Controller

**Files:**
- Create: `playcanvasViewer/camera_controller.h`
- Create: `playcanvasViewer/camera_controller.cpp`
- Modify: `playcanvasViewer/CMakeLists.txt`
- Test: compile `playcanvas_viewer_camera`

- [ ] **Step 1: Create `playcanvasViewer/camera_controller.h`**

```cpp
#pragma once

#include "gsplat_data_adapter.h"

#include <Eigen/Core>

namespace playcanvas_viewer {

struct CameraInputState {
  bool forward = false;
  bool backward = false;
  bool left = false;
  bool right = false;
  bool down = false;
  bool up = false;
};

struct SceneBounds {
  Eigen::Vector3f center{0.0f, 0.0f, 0.0f};
  float radius = 1.0f;
};

class CameraController {
 public:
  void resetToBounds(const SceneBounds& bounds);
  void orbit(float deltaX, float deltaY);
  void dolly(float amount);
  void fly(const CameraInputState& input, float deltaSeconds);
  void resize(int width, int height);

  Eigen::Matrix4f viewMatrix() const;
  Eigen::Matrix4f projectionMatrix() const;
  Eigen::Vector3f position() const { return position_; }
  Eigen::Vector3f forward() const;
  float nearPlane() const { return nearPlane_; }
  float farPlane() const { return farPlane_; }
  bool changed() const { return changed_; }
  void clearChanged() { changed_ = false; }

 private:
  Eigen::Vector3f position_{0.0f, 0.0f, 3.0f};
  Eigen::Vector3f target_{0.0f, 0.0f, 0.0f};
  Eigen::Vector3f up_{0.0f, 1.0f, 0.0f};
  int width_ = 1280;
  int height_ = 720;
  float fovYRadians_ = 60.0f * 3.14159265358979323846f / 180.0f;
  float nearPlane_ = 0.01f;
  float farPlane_ = 1000.0f;
  float flySpeed_ = 3.0f;
  bool changed_ = true;
};

SceneBounds computeSceneBounds(const GSplatRenderData& data);

}  // namespace playcanvas_viewer
```

- [ ] **Step 2: Create `playcanvasViewer/camera_controller.cpp`**

```cpp
#include "camera_controller.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>

namespace playcanvas_viewer {
namespace {

Eigen::Matrix4f makeLookAt(const Eigen::Vector3f& eye, const Eigen::Vector3f& target, const Eigen::Vector3f& up) {
  const Eigen::Vector3f f = (target - eye).normalized();
  const Eigen::Vector3f s = f.cross(up).normalized();
  const Eigen::Vector3f u = s.cross(f);

  Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
  result(0, 0) = s.x();
  result(0, 1) = s.y();
  result(0, 2) = s.z();
  result(1, 0) = u.x();
  result(1, 1) = u.y();
  result(1, 2) = u.z();
  result(2, 0) = -f.x();
  result(2, 1) = -f.y();
  result(2, 2) = -f.z();
  result(0, 3) = -s.dot(eye);
  result(1, 3) = -u.dot(eye);
  result(2, 3) = f.dot(eye);
  return result;
}

Eigen::Matrix4f makePerspective(float fovYRadians, float aspect, float nearPlane, float farPlane) {
  const float f = 1.0f / std::tan(fovYRadians * 0.5f);
  Eigen::Matrix4f result = Eigen::Matrix4f::Zero();
  result(0, 0) = f / aspect;
  result(1, 1) = f;
  result(2, 2) = (farPlane + nearPlane) / (nearPlane - farPlane);
  result(2, 3) = (2.0f * farPlane * nearPlane) / (nearPlane - farPlane);
  result(3, 2) = -1.0f;
  return result;
}

}  // namespace

SceneBounds computeSceneBounds(const GSplatRenderData& data) {
  SceneBounds bounds;
  if (data.empty()) {
    return bounds;
  }

  Eigen::Vector3f minPoint(data.centers[0].x, data.centers[0].y, data.centers[0].z);
  Eigen::Vector3f maxPoint = minPoint;

  for (size_t i = 0; i < data.size(); ++i) {
    const auto& center = data.centers[i];
    const auto& scale = data.scales[i];
    const float extent = 2.0f * std::max({scale.x, scale.y, scale.z});
    const Eigen::Vector3f point(center.x, center.y, center.z);
    minPoint = minPoint.cwiseMin(point - Eigen::Vector3f::Constant(extent));
    maxPoint = maxPoint.cwiseMax(point + Eigen::Vector3f::Constant(extent));
  }

  bounds.center = (minPoint + maxPoint) * 0.5f;
  bounds.radius = std::max(0.01f, (maxPoint - minPoint).norm() * 0.5f);
  return bounds;
}

void CameraController::resetToBounds(const SceneBounds& bounds) {
  target_ = bounds.center;
  const float distance = std::max(bounds.radius * 2.5f, 1.0f);
  position_ = target_ + Eigen::Vector3f(0.0f, 0.0f, distance);
  nearPlane_ = std::max(bounds.radius * 0.0005f, 0.001f);
  farPlane_ = std::max(bounds.radius * 32.0f, 10.0f);
  changed_ = true;
}

void CameraController::orbit(float deltaX, float deltaY) {
  Eigen::Vector3f offset = position_ - target_;
  const float distance = std::max(offset.norm(), 0.001f);
  offset.normalize();

  const Eigen::AngleAxisf yaw(-deltaX * 0.005f, up_);
  const Eigen::Vector3f right = forward().cross(up_).normalized();
  const Eigen::AngleAxisf pitch(-deltaY * 0.005f, right);
  offset = yaw * pitch * offset;
  position_ = target_ + offset.normalized() * distance;
  changed_ = true;
}

void CameraController::dolly(float amount) {
  Eigen::Vector3f offset = position_ - target_;
  const float scale = std::exp(-amount * 0.1f);
  position_ = target_ + offset * scale;
  changed_ = true;
}

void CameraController::fly(const CameraInputState& input, float deltaSeconds) {
  const Eigen::Vector3f f = forward();
  const Eigen::Vector3f r = f.cross(up_).normalized();
  Eigen::Vector3f movement = Eigen::Vector3f::Zero();
  movement += f * (static_cast<float>(input.forward) - static_cast<float>(input.backward));
  movement += r * (static_cast<float>(input.right) - static_cast<float>(input.left));
  movement += up_ * (static_cast<float>(input.up) - static_cast<float>(input.down));
  if (movement.squaredNorm() <= 0.0f) {
    return;
  }
  movement.normalize();
  movement *= flySpeed_ * std::max(deltaSeconds, 0.0f);
  position_ += movement;
  target_ += movement;
  changed_ = true;
}

void CameraController::resize(int width, int height) {
  width_ = std::max(width, 1);
  height_ = std::max(height, 1);
  changed_ = true;
}

Eigen::Matrix4f CameraController::viewMatrix() const { return makeLookAt(position_, target_, up_); }

Eigen::Matrix4f CameraController::projectionMatrix() const {
  return makePerspective(fovYRadians_, static_cast<float>(width_) / static_cast<float>(height_), nearPlane_, farPlane_);
}

Eigen::Vector3f CameraController::forward() const { return (target_ - position_).normalized(); }

}  // namespace playcanvas_viewer
```

- [ ] **Step 3: Add a camera library target**

Append to `playcanvasViewer/CMakeLists.txt`:

```cmake
add_library(playcanvas_viewer_camera STATIC
    camera_controller.cpp
)

target_include_directories(playcanvas_viewer_camera
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(playcanvas_viewer_camera
    PUBLIC
        playcanvas_viewer_adapter
        Eigen3::Eigen
)
```

- [ ] **Step 4: Build the camera target**

Run:

```bash
cmake --build build/playcanvas-viewer-plan --target playcanvas_viewer_camera -j
```

Expected: `Built target playcanvas_viewer_camera`.

- [ ] **Step 5: Commit the camera controller**

```bash
git add playcanvasViewer/CMakeLists.txt playcanvasViewer/camera_controller.h playcanvasViewer/camera_controller.cpp
git commit -m "feat: add PlayCanvas viewer camera controller"
```

### Task 4: Add PlayCanvas-Style Shader Sources

**Files:**
- Create: `playcanvasViewer/gsplat_shader_sources.h`
- Test: used by renderer build in Task 5

- [ ] **Step 1: Create `playcanvasViewer/gsplat_shader_sources.h`**

```cpp
#pragma once

namespace playcanvas_viewer {

inline constexpr const char* kGSplatVertexShader = R"GLSL(
#version 330 core

layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec3 aCenter;
layout(location = 2) in vec4 aRotation;  // w, x, y, z
layout(location = 3) in vec3 aScale;
layout(location = 4) in vec4 aColor;

uniform mat4 uView;
uniform mat4 uProjection;
uniform vec4 uViewport;       // width, height, 1 / width, 1 / height
uniform vec4 uCameraParams;   // 1 / far, far, near, isOrtho
uniform float uMinPixelSize;
uniform float uSizeScale;

out vec2 vUv;
out vec4 vColor;

mat3 quatToMat3(vec4 q) {
  float x = q.y;
  float y = q.z;
  float z = q.w;
  float w = q.x;

  float x2 = x + x;
  float y2 = y + y;
  float z2 = z + z;
  float xx = x * x2;
  float xy = x * y2;
  float xz = x * z2;
  float yy = y * y2;
  float yz = y * z2;
  float zz = z * z2;
  float wx = w * x2;
  float wy = w * y2;
  float wz = w * z2;

  return mat3(
      1.0 - (yy + zz), xy + wz, xz - wy,
      xy - wz, 1.0 - (xx + zz), yz + wx,
      xz + wy, yz - wx, 1.0 - (xx + yy));
}

void computeCovariance(vec4 rotation, vec3 scale, out vec3 covA, out vec3 covB) {
  mat3 rot = quatToMat3(rotation);
  mat3 M = transpose(mat3(
      scale.x * rot[0],
      scale.y * rot[1],
      scale.z * rot[2]));

  covA = vec3(dot(M[0], M[0]), dot(M[0], M[1]), dot(M[0], M[2]));
  covB = vec3(dot(M[1], M[1]), dot(M[1], M[2]), dot(M[2], M[2]));
}

bool initCorner(vec3 viewCenter, mat3 modelView3, vec4 centerProj, float projMat00, vec3 covA, vec3 covB,
                out vec2 offset, out vec2 uv) {
  mat3 Vrk = mat3(
      covA.x, covA.y, covA.z,
      covA.y, covB.x, covB.y,
      covA.z, covB.y, covB.z);

  float focal = uViewport.x * projMat00;
  vec3 vp = uCameraParams.w == 1.0 ? vec3(0.0, 0.0, 1.0) : viewCenter;
  float J1 = focal / vp.z;
  vec2 J2 = -J1 / vp.z * vp.xy;
  mat3 J = mat3(
      J1, 0.0, J2.x,
      0.0, J1, J2.y,
      0.0, 0.0, 0.0);

  mat3 W = transpose(modelView3);
  mat3 T = W * J;
  mat3 cov = transpose(T) * Vrk * T;

  float diagonal1 = cov[0][0] + 0.3;
  float offDiagonal = cov[0][1];
  float diagonal2 = cov[1][1] + 0.3;

  float mid = 0.5 * (diagonal1 + diagonal2);
  float radius = length(vec2((diagonal1 - diagonal2) * 0.5, offDiagonal));
  float lambda1 = mid + radius;
  float lambda2 = max(mid - radius, 0.1);

  float vmin = min(1024.0, min(uViewport.x, uViewport.y));
  float l1 = 2.0 * min(sqrt(2.0 * lambda1), vmin) * uSizeScale;
  float l2 = 2.0 * min(sqrt(2.0 * lambda2), vmin) * uSizeScale;

  if (max(l1, l2) < uMinPixelSize) {
    return false;
  }

  vec2 c = centerProj.ww * uViewport.zw;
  if (any(greaterThan(abs(centerProj.xy) - vec2(max(l1, l2)) * c, centerProj.ww))) {
    return false;
  }

  vec2 diagonalVector = normalize(vec2(offDiagonal, lambda1 - diagonal1));
  vec2 v1 = l1 * diagonalVector;
  vec2 v2 = l2 * vec2(diagonalVector.y, -diagonalVector.x);

  offset = (aCorner.x * v1 + aCorner.y * v2) * c;
  uv = aCorner;
  return true;
}

void main() {
  vec4 centerView4 = uView * vec4(aCenter, 1.0);
  if (uCameraParams.w != 1.0 && centerView4.z > 0.0) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vUv = vec2(2.0);
    vColor = vec4(0.0);
    return;
  }

  vec4 centerProj = uProjection * centerView4;
  centerProj.z = clamp(centerProj.z, -abs(centerProj.w), abs(centerProj.w));

  vec3 covA;
  vec3 covB;
  computeCovariance(aRotation, aScale, covA, covB);

  vec2 offset;
  vec2 uv;
  if (!initCorner(centerView4.xyz / centerView4.w, mat3(uView), centerProj, uProjection[0][0], covA, covB, offset, uv)) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vUv = vec2(2.0);
    vColor = vec4(0.0);
    return;
  }

  gl_Position = centerProj + vec4(offset, 0.0, 0.0);
  vUv = uv;
  vColor = aColor;
}
)GLSL";

inline constexpr const char* kGSplatFragmentShader = R"GLSL(
#version 330 core

in vec2 vUv;
in vec4 vColor;

uniform float uGlobalOpacity;
uniform float uAlphaDiscardThreshold;

out vec4 fragColor;

const float EXP4 = exp(-4.0);
const float INV_EXP4 = 1.0 / (1.0 - EXP4);

float normExp(float x) {
  return (exp(x * -4.0) - EXP4) * INV_EXP4;
}

void main() {
  float A = dot(vUv, vUv);
  if (A > 1.0) {
    discard;
  }

  float alpha = normExp(A) * clamp(vColor.a * uGlobalOpacity, 0.0, 1.0);
  if (alpha < uAlphaDiscardThreshold) {
    discard;
  }

  fragColor = vec4(max(vColor.rgb, vec3(0.0)) * alpha, alpha);
}
)GLSL";

}  // namespace playcanvas_viewer
```

- [ ] **Step 2: Commit shader sources**

```bash
git add playcanvasViewer/gsplat_shader_sources.h
git commit -m "feat: add PlayCanvas-style gsplat shaders"
```

### Task 5: Implement The OpenGL Renderer

**Files:**
- Create: `playcanvasViewer/gsplat_renderer.h`
- Create: `playcanvasViewer/gsplat_renderer.cpp`
- Modify: `playcanvasViewer/CMakeLists.txt`
- Test: build `playcanvas_viewer_renderer`

- [ ] **Step 1: Create `playcanvasViewer/gsplat_renderer.h`**

```cpp
#pragma once

#include "camera_controller.h"
#include "gsplat_data_adapter.h"

#include <Eigen/Core>
#include <GL/glew.h>

#include <vector>

namespace playcanvas_viewer {

struct GSplatRenderOptions {
  float globalOpacity = 1.0f;
  float sizeScale = 1.0f;
  float minPixelSize = 2.0f;
  float alphaDiscardThreshold = 1.0f / 255.0f;
  bool sortBackToFront = true;
};

class GSplatRenderer {
 public:
  GSplatRenderer();
  ~GSplatRenderer();

  GSplatRenderer(const GSplatRenderer&) = delete;
  GSplatRenderer& operator=(const GSplatRenderer&) = delete;

  void setData(const GSplatRenderData& data);
  void render(const CameraController& camera, int width, int height, const GSplatRenderOptions& options);

 private:
  struct InstanceData {
    Vec3f center;
    Vec4f rotation;
    Vec3f scale;
    Vec4f color;
  };

  void ensureProgram();
  void ensureBuffers();
  void updateSortedInstances(const CameraController& camera, bool sortBackToFront);
  void uploadInstances();
  GLint uniformLocation(const char* name) const;

  GSplatRenderData data_;
  std::vector<uint32_t> order_;
  std::vector<InstanceData> sortedInstances_;

  GLuint vao_ = 0;
  GLuint cornerBuffer_ = 0;
  GLuint instanceBuffer_ = 0;
  GLuint program_ = 0;
  bool instancesDirty_ = true;
};

}  // namespace playcanvas_viewer
```

- [ ] **Step 2: Create `playcanvasViewer/gsplat_renderer.cpp`**

```cpp
#include "gsplat_renderer.h"

#include "gsplat_shader_sources.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace playcanvas_viewer {
namespace {

GLuint compileShader(GLenum type, const char* source) {
  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  GLint success = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (success == GL_TRUE) {
    return shader;
  }

  GLint logLength = 0;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
  std::string log(static_cast<size_t>(std::max(logLength, 1)), '\0');
  glGetShaderInfoLog(shader, logLength, nullptr, log.data());
  glDeleteShader(shader);
  throw std::runtime_error("PlaycanvasViewer shader compile failed: " + log);
}

GLuint buildProgram() {
  const GLuint vertexShader = compileShader(GL_VERTEX_SHADER, kGSplatVertexShader);
  const GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, kGSplatFragmentShader);
  const GLuint program = glCreateProgram();
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);
  glLinkProgram(program);
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  GLint success = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (success == GL_TRUE) {
    return program;
  }

  GLint logLength = 0;
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
  std::string log(static_cast<size_t>(std::max(logLength, 1)), '\0');
  glGetProgramInfoLog(program, logLength, nullptr, log.data());
  glDeleteProgram(program);
  throw std::runtime_error("PlaycanvasViewer shader link failed: " + log);
}

void setMat4(GLint location, const Eigen::Matrix4f& matrix) {
  glUniformMatrix4fv(location, 1, GL_TRUE, matrix.data());
}

}  // namespace

GSplatRenderer::GSplatRenderer() = default;

GSplatRenderer::~GSplatRenderer() {
  if (instanceBuffer_ != 0) glDeleteBuffers(1, &instanceBuffer_);
  if (cornerBuffer_ != 0) glDeleteBuffers(1, &cornerBuffer_);
  if (vao_ != 0) glDeleteVertexArrays(1, &vao_);
  if (program_ != 0) glDeleteProgram(program_);
}

void GSplatRenderer::setData(const GSplatRenderData& data) {
  data_ = data;
  order_.resize(data_.size());
  for (size_t i = 0; i < order_.size(); ++i) {
    order_[i] = static_cast<uint32_t>(i);
  }
  sortedInstances_.resize(data_.size());
  instancesDirty_ = true;
}

void GSplatRenderer::ensureProgram() {
  if (program_ == 0) {
    program_ = buildProgram();
  }
}

void GSplatRenderer::ensureBuffers() {
  if (vao_ != 0) {
    return;
  }

  const float corners[] = {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);

  glGenBuffers(1, &cornerBuffer_);
  glBindBuffer(GL_ARRAY_BUFFER, cornerBuffer_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(corners), corners, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);

  glGenBuffers(1, &instanceBuffer_);
  glBindBuffer(GL_ARRAY_BUFFER, instanceBuffer_);
  glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

  const GLsizei stride = sizeof(InstanceData);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(InstanceData, center)));
  glVertexAttribDivisor(1, 1);

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(InstanceData, rotation)));
  glVertexAttribDivisor(2, 1);

  glEnableVertexAttribArray(3);
  glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(InstanceData, scale)));
  glVertexAttribDivisor(3, 1);

  glEnableVertexAttribArray(4);
  glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(InstanceData, color)));
  glVertexAttribDivisor(4, 1);

  glBindVertexArray(0);
}

void GSplatRenderer::updateSortedInstances(const CameraController& camera, bool sortBackToFront) {
  if (sortBackToFront) {
    const Eigen::Vector3f cameraPosition = camera.position();
    const Eigen::Vector3f cameraForward = camera.forward();
    std::sort(order_.begin(), order_.end(), [&](uint32_t lhs, uint32_t rhs) {
      const auto& lc = data_.centers[lhs];
      const auto& rc = data_.centers[rhs];
      const float ld = (Eigen::Vector3f(lc.x, lc.y, lc.z) - cameraPosition).dot(cameraForward);
      const float rd = (Eigen::Vector3f(rc.x, rc.y, rc.z) - cameraPosition).dot(cameraForward);
      return ld > rd;
    });
  }

  for (size_t sortedIndex = 0; sortedIndex < order_.size(); ++sortedIndex) {
    const uint32_t sourceIndex = order_[sortedIndex];
    sortedInstances_[sortedIndex] = {data_.centers[sourceIndex], data_.rotations[sourceIndex], data_.scales[sourceIndex],
                                     data_.colors[sourceIndex]};
  }
  instancesDirty_ = true;
}

void GSplatRenderer::uploadInstances() {
  if (!instancesDirty_) {
    return;
  }
  glBindBuffer(GL_ARRAY_BUFFER, instanceBuffer_);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sortedInstances_.size() * sizeof(InstanceData)),
               sortedInstances_.data(), GL_DYNAMIC_DRAW);
  instancesDirty_ = false;
}

GLint GSplatRenderer::uniformLocation(const char* name) const { return glGetUniformLocation(program_, name); }

void GSplatRenderer::render(const CameraController& camera, int width, int height, const GSplatRenderOptions& options) {
  if (data_.empty()) {
    return;
  }

  ensureProgram();
  ensureBuffers();
  updateSortedInstances(camera, options.sortBackToFront);
  uploadInstances();

  glUseProgram(program_);
  setMat4(uniformLocation("uView"), camera.viewMatrix());
  setMat4(uniformLocation("uProjection"), camera.projectionMatrix());
  glUniform4f(uniformLocation("uViewport"), static_cast<float>(width), static_cast<float>(height), 1.0f / width,
              1.0f / height);
  glUniform4f(uniformLocation("uCameraParams"), 1.0f / camera.farPlane(), camera.farPlane(), camera.nearPlane(), 0.0f);
  glUniform1f(uniformLocation("uMinPixelSize"), options.minPixelSize);
  glUniform1f(uniformLocation("uSizeScale"), options.sizeScale);
  glUniform1f(uniformLocation("uGlobalOpacity"), options.globalOpacity);
  glUniform1f(uniformLocation("uAlphaDiscardThreshold"), options.alphaDiscardThreshold);

  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  glDisable(GL_CULL_FACE);
  glDepthMask(GL_FALSE);
  glEnable(GL_DEPTH_TEST);

  glBindVertexArray(vao_);
  glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(sortedInstances_.size()));
  glBindVertexArray(0);
}

}  // namespace playcanvas_viewer
```

- [ ] **Step 3: Add the renderer target**

Append to `playcanvasViewer/CMakeLists.txt`:

```cmake
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

add_library(playcanvas_viewer_renderer STATIC
    gsplat_renderer.cpp
)

target_include_directories(playcanvas_viewer_renderer
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(playcanvas_viewer_renderer
    PUBLIC
        playcanvas_viewer_adapter
        playcanvas_viewer_camera
        GLEW::GLEW
        OpenGL::GL
)
```

- [ ] **Step 4: Build the renderer target**

Run:

```bash
cmake --build build/playcanvas-viewer-plan --target playcanvas_viewer_renderer -j
```

Expected: `Built target playcanvas_viewer_renderer`.

- [ ] **Step 5: Commit the renderer**

```bash
git add playcanvasViewer/CMakeLists.txt playcanvasViewer/gsplat_renderer.h playcanvasViewer/gsplat_renderer.cpp
git commit -m "feat: add OpenGL gsplat renderer"
```

### Task 6: Add The PlaycanvasViewer Executable

**Files:**
- Create: `playcanvasViewer/main.cpp`
- Modify: `playcanvasViewer/CMakeLists.txt`
- Test: build `PlaycanvasViewer`

- [ ] **Step 1: Create `playcanvasViewer/main.cpp`**

```cpp
#include "camera_controller.h"
#include "gsplat_data_adapter.h"
#include "gsplat_renderer.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <splat/splat.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct AppOptions {
  std::filesystem::path input;
  int width = 1280;
  int height = 720;
  size_t maxSplats = 0;
  bool sort = true;
};

struct AppState {
  playcanvas_viewer::CameraController camera;
  playcanvas_viewer::GSplatRenderer renderer;
  playcanvas_viewer::GSplatRenderOptions renderOptions;
  bool dragging = false;
  double lastMouseX = 0.0;
  double lastMouseY = 0.0;
  playcanvas_viewer::SceneBounds bounds;
};

std::string lowerExtension(const std::filesystem::path& path) {
  std::string ext = path.extension().u8string();
  for (char& c : ext) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return ext;
}

std::unique_ptr<splat::DataTable> readInput(const std::filesystem::path& filename) {
  const std::string ext = lowerExtension(filename);
  if (ext == ".ply") return splat::readPly(filename);
  if (ext == ".splat") return splat::readSplat(filename);
  if (ext == ".sog") return splat::readSog(filename, filename);
  if (ext == ".spz") return splat::readSpz(filename);
  if (ext == ".ksplat") return splat::readKsplat(filename);
  throw std::runtime_error("Unsupported input format: " + ext);
}

AppOptions parseOptions(int argc, char** argv) {
  AppOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.rfind("--width=", 0) == 0) {
      options.width = std::max(1, std::stoi(arg.substr(8)));
    } else if (arg.rfind("--height=", 0) == 0) {
      options.height = std::max(1, std::stoi(arg.substr(9)));
    } else if (arg.rfind("--max-splats=", 0) == 0) {
      options.maxSplats = static_cast<size_t>(std::stoull(arg.substr(13)));
    } else if (arg == "--no-sort") {
      options.sort = false;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: PlaycanvasViewer [--width=N] [--height=N] [--max-splats=N] [--no-sort] <input-file>\n";
      std::exit(0);
    } else if (options.input.empty()) {
      options.input = std::filesystem::u8path(arg);
    } else {
      throw std::runtime_error("Unexpected argument: " + arg);
    }
  }

  if (options.input.empty()) {
    throw std::runtime_error("Missing input file. Use --help for usage.");
  }
  return options;
}

playcanvas_viewer::CameraInputState makeInputState(GLFWwindow* window) {
  playcanvas_viewer::CameraInputState input;
  input.forward = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
  input.backward = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
  input.left = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
  input.right = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;
  input.down = glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS;
  input.up = glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS;
  return input;
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
  auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
  state->camera.resize(width, height);
  glViewport(0, 0, width, height);
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int) {
  auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    state->dragging = action == GLFW_PRESS;
    glfwGetCursorPos(window, &state->lastMouseX, &state->lastMouseY);
  }
}

void cursorPositionCallback(GLFWwindow* window, double x, double y) {
  auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (!state->dragging) {
    state->lastMouseX = x;
    state->lastMouseY = y;
    return;
  }
  state->camera.orbit(static_cast<float>(x - state->lastMouseX), static_cast<float>(y - state->lastMouseY));
  state->lastMouseX = x;
  state->lastMouseY = y;
}

void scrollCallback(GLFWwindow* window, double, double yoffset) {
  auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
  state->camera.dolly(static_cast<float>(yoffset));
}

void keyCallback(GLFWwindow* window, int key, int, int action, int) {
  auto* state = static_cast<AppState*>(glfwGetWindowUserPointer(window));
  if (action != GLFW_PRESS) {
    return;
  }
  if (key == GLFW_KEY_ESCAPE) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  } else if (key == GLFW_KEY_R) {
    state->camera.resetToBounds(state->bounds);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const AppOptions options = parseOptions(argc, argv);
    auto table = readInput(options.input);
    if (!table || table->getNumRows() == 0) {
      throw std::runtime_error("Input did not contain any splats");
    }

    auto data = playcanvas_viewer::adaptDataTableToGSplat(*table, options.maxSplats);
    if (data.empty()) {
      throw std::runtime_error("No renderable splats after adaptation");
    }

    if (!glfwInit()) {
      throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(options.width, options.height, "PlaycanvasViewer", nullptr, nullptr);
    if (!window) {
      glfwTerminate();
      throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glewExperimental = GL_TRUE;
    const GLenum glewStatus = glewInit();
    if (glewStatus != GLEW_OK) {
      throw std::runtime_error(reinterpret_cast<const char*>(glewGetErrorString(glewStatus)));
    }

    AppState state;
    state.renderOptions.sortBackToFront = options.sort;
    state.bounds = playcanvas_viewer::computeSceneBounds(data);
    state.camera.resize(options.width, options.height);
    state.camera.resetToBounds(state.bounds);
    state.renderer.setData(data);

    glfwSetWindowUserPointer(window, &state);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);

    auto lastFrame = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window)) {
      const auto now = std::chrono::steady_clock::now();
      const float deltaSeconds = std::chrono::duration<float>(now - lastFrame).count();
      lastFrame = now;

      glfwPollEvents();
      state.camera.fly(makeInputState(window), deltaSeconds);

      int width = 0;
      int height = 0;
      glfwGetFramebufferSize(window, &width, &height);
      glViewport(0, 0, width, height);
      glClearColor(0.03f, 0.035f, 0.045f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      state.renderer.render(state.camera, std::max(width, 1), std::max(height, 1), state.renderOptions);

      glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "PlaycanvasViewer error: " << error.what() << "\n";
    return 1;
  }
}
```

- [ ] **Step 2: Add the executable target**

Append to `playcanvasViewer/CMakeLists.txt`:

```cmake
find_package(glfw3 CONFIG REQUIRED)

add_executable(PlaycanvasViewer
    main.cpp
)

target_link_libraries(PlaycanvasViewer
    PRIVATE
        playcanvas_viewer_adapter
        playcanvas_viewer_camera
        playcanvas_viewer_renderer
        glfw
        GLEW::GLEW
        OpenGL::GL
)

message(STATUS "PlaycanvasViewer: Building independent PlayCanvas-style OpenGL viewer")
```

- [ ] **Step 3: Build the executable**

Run:

```bash
cmake -S . -B build/playcanvas-viewer-plan -DBUILD_SPLAT_PLAYCANVAS_VIEWER=ON -DBUILD_SPLAT_TESTS=ON
cmake --build build/playcanvas-viewer-plan --target PlaycanvasViewer -j
```

Expected: `Built target PlaycanvasViewer`.

- [ ] **Step 4: Commit the executable**

```bash
git add playcanvasViewer/CMakeLists.txt playcanvasViewer/main.cpp
git commit -m "feat: add PlayCanvas viewer executable"
```

### Task 7: Run Runtime Smoke Validation

**Files:**
- No required source changes unless the build/runtime check reveals a focused issue
- Test: local sample command

- [ ] **Step 1: Find a local renderable sample**

Run:

```bash
find data -type f \( -name '*.ply' -o -name '*.splat' -o -name '*.sog' -o -name '*.spz' -o -name '*.ksplat' \) | head -n 10
```

Expected: at least one input file path. If none exists, use a known local input path supplied by the user.

- [ ] **Step 2: Launch the viewer with a splat limit first**

Run, replacing `<input-file>` with a real result from Step 1:

```bash
./build/playcanvas-viewer-plan/playcanvasViewer/PlaycanvasViewer --max-splats=200000 <input-file>
```

Expected: a window opens and shows visible Gaussian ellipses. Mouse orbit, mouse wheel dolly, `W/A/S/D/Q/E`, `R`, and `Esc` work.

- [ ] **Step 3: Launch the viewer without the splat limit**

Run:

```bash
./build/playcanvas-viewer-plan/playcanvasViewer/PlaycanvasViewer <input-file>
```

Expected: the full file opens. If sorting is slow, record the behavior and try:

```bash
./build/playcanvas-viewer-plan/playcanvasViewer/PlaycanvasViewer --no-sort <input-file>
```

- [ ] **Step 4: Capture any runtime-driven fix as a small patch**

If the runtime test exposes a specific shader, camera, or GL state issue, fix only that issue and rerun:

```bash
cmake --build build/playcanvas-viewer-plan --target PlaycanvasViewer -j
./build/playcanvas-viewer-plan/playcanvasViewer/PlaycanvasViewer --max-splats=200000 <input-file>
```

Expected: the previously observed runtime issue is resolved.

- [ ] **Step 5: Commit runtime follow-ups if any were needed**

If source files changed in Step 4, run:

```bash
git add playcanvasViewer
git commit -m "fix: stabilize PlayCanvas viewer runtime path"
```

Skip this commit if no source changes were needed.

### Task 8: Final Verification And Documentation Note

**Files:**
- Modify: `README.md` only if the user wants the new experimental tool documented in the public README immediately
- Test: build and adapter test

- [ ] **Step 1: Run final build and adapter test**

Run:

```bash
cmake -S . -B build/playcanvas-viewer-plan -DBUILD_SPLAT_PLAYCANVAS_VIEWER=ON -DBUILD_SPLAT_TESTS=ON
cmake --build build/playcanvas-viewer-plan --target PlaycanvasViewer PlaycanvasViewerAdapterTests -j
ctest --test-dir build/playcanvas-viewer-plan --output-on-failure -R PlaycanvasViewerAdapterTests
```

Expected: `PlaycanvasViewer` builds and `PlaycanvasViewerAdapterTests` passes.

- [ ] **Step 2: Inspect that VTK viewer files were not touched**

Run:

```bash
git diff --name-only HEAD -- include/splat/visualization src/visualization viewer
```

Expected: no output.

- [ ] **Step 3: Document the experimental build command if requested**

If public docs should mention the tool now, add this short section to `README.md`:

````md
### Experimental PlayCanvas-Style Viewer

SplatLib includes an optional OpenGL viewer that follows the PlayCanvas GSplat rendering path and does not depend on VTK:

```bash
cmake -S . -B build/playcanvas-viewer -DBUILD_SPLAT_PLAYCANVAS_VIEWER=ON
cmake --build build/playcanvas-viewer --target PlaycanvasViewer -j
./build/playcanvas-viewer/playcanvasViewer/PlaycanvasViewer <input-file>
```
````

Skip this step if the tool should remain undocumented until visual verification is complete.

- [ ] **Step 4: Commit documentation if it changed**

```bash
git add README.md
git commit -m "docs: mention experimental PlayCanvas viewer"
```

Skip this commit if `README.md` was not changed.

## Self-Review

- Spec coverage: the plan adds an independent `playcanvasViewer/` tool, preserves `DataTable` as input, mirrors PlayCanvas decode and covariance shader math, avoids VTK changes, and includes adapter/build/runtime validation.
- Placeholder scan: no task depends on unspecified files or undefined helper names; optional documentation and runtime-fix commits include explicit skip conditions.
- Type consistency: `GSplatRenderData`, `CameraController`, `GSplatRenderer`, `GSplatRenderOptions`, and target names are consistent across tasks.
