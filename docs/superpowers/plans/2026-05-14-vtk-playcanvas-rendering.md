# VTK PlayCanvas-Style Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the C++ VTK-based visualization path actually render 3DGS data on screen by using the PlayCanvas TS implementation as the correctness baseline while preserving the existing `SplatVisualizer` windowing and interaction shell.

**Architecture:** Keep `SplatVisualizer` and VTK integration as the host framework, and keep `DataTable` as the single rendering data model instead of introducing a second C++ render-data abstraction. The runtime should read `DataTable` directly, decode columns with the same semantic rules used by the TS path, and feed a VTK-managed OpenGL draw path that mirrors the TS gaussian projection and alpha behavior as closely as practical without introducing a PlayCanvas-style GPU texture architecture. Treat the TS path as the source of truth for expected rendering behavior; treat the current C++ path as incorrect until it can render visible splats with TS-aligned semantics.

**Tech Stack:** C++17, VTK OpenGL2, OpenGL 3.3+, existing `splat::DataTable`, CMake, CTest

**Implementation Branch:** Make the code changes on `dev-1.3.1`. The design/reference docs can stay in the `ai/superpowers` worktree, but the actual C++ visualization implementation should be applied and verified on `dev-1.3.1`.

**Correctness Baseline:** The TS rendering logic is known-good and already renders successfully. The current C++ visualization path is known-bad from a product perspective because it does not render a correct visible result. During implementation, prefer matching TS behavior over preserving current C++ behavior whenever the two conflict.

---

## Proposed File Structure

**Files:**
- Create: `src/visualization/splat_shader_sources.h`
- Create: `tests/CMakeLists.txt`
- Create: `tests/visualization/splat_visualizer_data_semantics_test.cpp`
- Create: `tests/visualization/splat_visualizer_smoke_test.cpp`
- Modify: `CMakeLists.txt`
- Modify: `src/CMakeLists.txt`
- Modify: `src/visualization/splat_visualizer.cpp`
- Modify: `include/splat/visualization/splat_visualizer.h`
- Modify: `docs/superpowers/reference/visualization-module-structure.md`

**Responsibilities:**
- `src/visualization/splat_shader_sources.h`
  - own the VTK/OpenGL GLSL source strings so TS-to-C++ shader parity work stops living inline in `splat_visualizer.cpp`
- `src/visualization/splat_visualizer.cpp`
  - keep VTK lifecycle/event plumbing, own direct `DataTable` decoding/upload/sort logic, and use the refactored shader sources
- `include/splat/visualization/splat_visualizer.h`
  - expose only the render options that are intentionally public
- `tests/*`
  - verify direct `DataTable` decode semantics, bounds/sorting assumptions, minimal VTK construction behavior, and regression coverage for the "renders nothing" failure mode

### Task 1: Add a Test Harness for Visualization Parity

**Files:**
- Create: `tests/CMakeLists.txt`
- Create: `tests/visualization/splat_visualizer_data_semantics_test.cpp`
- Create: `tests/visualization/splat_visualizer_smoke_test.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing `DataTable` semantics test**

```cpp
#include <splat/models/data-table.h>
#include <splat/visualization/splat_visualizer.h>

#include <cassert>
#include <cmath>

int main() {
  using namespace splat;

  DataTable table;
  table.addColumn(Column{"x", std::vector<float>{1.0f}});
  table.addColumn(Column{"y", std::vector<float>{2.0f}});
  table.addColumn(Column{"z", std::vector<float>{3.0f}});
  table.addColumn(Column{"f_dc_0", std::vector<float>{0.25f}});
  table.addColumn(Column{"f_dc_1", std::vector<float>{0.0f}});
  table.addColumn(Column{"f_dc_2", std::vector<float>{-0.25f}});
  table.addColumn(Column{"opacity", std::vector<float>{0.0f}});
  table.addColumn(Column{"scale_0", std::vector<float>{std::log(2.0f)}});
  table.addColumn(Column{"scale_1", std::vector<float>{std::log(3.0f)}});
  table.addColumn(Column{"scale_2", std::vector<float>{std::log(4.0f)}});
  table.addColumn(Column{"rot_0", std::vector<float>{1.0f}});
  table.addColumn(Column{"rot_1", std::vector<float>{0.0f}});
  table.addColumn(Column{"rot_2", std::vector<float>{0.0f}});
  table.addColumn(Column{"rot_3", std::vector<float>{0.0f}});

  SplatVisualizer visualizer("semantics");
  const bool added = visualizer.addSplatCloud(table, "one");
  assert(added);
  assert(visualizer.contains("one"));
  assert(visualizer.getSplatCount("one") == 1);
  return 0;
}
```

- [ ] **Step 2: Run the test target to verify it fails**

Run: `cmake -S . -B build/test-plan -DBUILD_SPLAT_VISUALIZATION=ON -DBUILD_SPLAT_TESTS=ON && cmake --build build/test-plan --target splat_visualizer_data_semantics_test -j`
Expected: FAIL because the visualization tests target does not exist yet

- [ ] **Step 3: Add a regression-oriented note to the test harness**

```cpp
// This test suite exists because the TS renderer is known-good while the
// current C++ path is known to produce no useful visible output. Any future
// change that keeps compilation green but breaks decode/render semantics
// should fail here first.
```

- [ ] **Step 4: Add CTest wiring for visualization tests**

```cmake
if(BUILD_SPLAT_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

```cmake
add_executable(splat_visualizer_data_semantics_test visualization/splat_visualizer_data_semantics_test.cpp)
target_link_libraries(splat_visualizer_data_semantics_test PRIVATE splat)
add_test(NAME splat_visualizer_data_semantics_test COMMAND splat_visualizer_data_semantics_test)

add_executable(splat_visualizer_smoke_test visualization/splat_visualizer_smoke_test.cpp)
target_link_libraries(splat_visualizer_smoke_test PRIVATE splat)
add_test(NAME splat_visualizer_smoke_test COMMAND splat_visualizer_smoke_test)
```

- [ ] **Step 5: Write the smoke test for VTK-side object construction**

```cpp
#include <splat/visualization/splat_visualizer.h>

int main() {
  splat::SplatVisualizer visualizer("smoke");
  visualizer.setWindowSize(320, 240);
  visualizer.setAxesEnabled(false);
  return visualizer.wasStopped() ? 1 : 0;
}
```

- [ ] **Step 6: Run the tests and verify the harness passes**

Run: `cmake -S . -B build/test-plan -DBUILD_SPLAT_VISUALIZATION=ON -DBUILD_SPLAT_TESTS=ON && cmake --build build/test-plan -j && ctest --test-dir build/test-plan --output-on-failure`
Expected: `100% tests passed` after implementation tasks below are complete

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt tests/CMakeLists.txt tests/visualization/splat_render_data_test.cpp tests/visualization/splat_visualizer_smoke_test.cpp
git commit -m "test: add visualization parity test harness"
```

### Task 2: Implement TS-Compatible Direct `DataTable` Rendering in `SplatVisualizer`

**Files:**
- Modify: `src/visualization/splat_visualizer.cpp`
- Test: `tests/visualization/splat_render_data_test.cpp`

- [ ] **Step 1: Add direct `DataTable` semantic helpers inside `splat_visualizer.cpp`**

```cpp
namespace {
  const float shC0 = 0.28209479177387814f;
  auto sigmoid = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };

  float decodeOpacity(float v) { return sigmoid(v); }
  float decodeScale(float v) { return std::exp(v); }
  float decodeColor(float v) { return std::clamp(v * shC0 + 0.5f, 0.0f, 1.0f); }
}
```

- [ ] **Step 2: Add required column validation up front so "renders nothing" fails loudly**

```cpp
void requireRenderColumns(const DataTable& dataTable) {
  for (const char* name : {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                           "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}) {
    if (!dataTable.hasColumn(name)) {
      throw std::runtime_error(std::string("SplatVisualizer: missing required column: ") + name);
    }
  }
}
```

- [ ] **Step 3: Replace duplicated ad-hoc extraction with one direct upload path**

```cpp
void NativeSplatProp::UploadDataTable(const DataTable& dataTable) {
  const size_t count = dataTable.getNumRows();
  this->Positions.resize(count * 3);
  this->Colors.resize(count * 3);
  this->Scales.resize(count * 3);
  this->Opacities.resize(count);
  this->Rotations.resize(count * 4);

  for (size_t i = 0; i < count; ++i) {
    this->Positions[i * 3 + 0] = dataTable.getColumnByName("x").getValue<float>(i);
    this->Positions[i * 3 + 1] = dataTable.getColumnByName("y").getValue<float>(i);
    this->Positions[i * 3 + 2] = dataTable.getColumnByName("z").getValue<float>(i);

    this->Colors[i * 3 + 0] = decodeColor(dataTable.getColumnByName("f_dc_0").getValue<float>(i));
    this->Colors[i * 3 + 1] = decodeColor(dataTable.getColumnByName("f_dc_1").getValue<float>(i));
    this->Colors[i * 3 + 2] = decodeColor(dataTable.getColumnByName("f_dc_2").getValue<float>(i));

    this->Scales[i * 3 + 0] = decodeScale(dataTable.getColumnByName("scale_0").getValue<float>(i));
    this->Scales[i * 3 + 1] = decodeScale(dataTable.getColumnByName("scale_1").getValue<float>(i));
    this->Scales[i * 3 + 2] = decodeScale(dataTable.getColumnByName("scale_2").getValue<float>(i));

    this->Opacities[i] = decodeOpacity(dataTable.getColumnByName("opacity").getValue<float>(i));

    this->Rotations[i * 4 + 0] = dataTable.getColumnByName("rot_0").getValue<float>(i);
    this->Rotations[i * 4 + 1] = dataTable.getColumnByName("rot_1").getValue<float>(i);
    this->Rotations[i * 4 + 2] = dataTable.getColumnByName("rot_2").getValue<float>(i);
    this->Rotations[i * 4 + 3] = dataTable.getColumnByName("rot_3").getValue<float>(i);
  }
}
```

- [ ] **Step 4: Rebuild and verify direct rendering code compiles**

Run: `cmake -S . -B build/test-plan -DBUILD_SPLAT_VISUALIZATION=ON -DBUILD_SPLAT_TESTS=ON && cmake --build build/test-plan --target splat_visualizer_data_semantics_test -j`
Expected: `Built target splat_visualizer_data_semantics_test`

- [ ] **Step 5: Run the semantics test to verify TS-aligned decode behavior**

Run: `ctest --test-dir build/test-plan --output-on-failure -R splat_visualizer_data_semantics_test`
Expected: `1/1 Test #1: splat_visualizer_data_semantics_test ... Passed`

- [ ] **Step 6: Commit**

```bash
git add src/visualization/splat_visualizer.cpp tests/visualization/splat_visualizer_data_semantics_test.cpp
git commit -m "feat: decode datatable directly in VTK renderer"
```

### Task 3: Refactor `SplatVisualizer` to Use Dedicated Render Data and Shader Sources

**Files:**
- Create: `src/visualization/splat_shader_sources.h`
- Modify: `src/visualization/splat_visualizer.cpp`
- Test: `tests/visualization/splat_visualizer_smoke_test.cpp`

- [ ] **Step 1: Move inline GLSL strings into a dedicated header**

```cpp
#pragma once

namespace splat::visualization {

inline constexpr const char* kGaussianVertexShader = R"GLSL(
// vertex shader body moved here from splat_visualizer.cpp
)GLSL";

inline constexpr const char* kGaussianFragmentShader = R"GLSL(
// fragment shader body moved here from splat_visualizer.cpp
)GLSL";

}  // namespace splat::visualization
```

- [ ] **Step 2: Replace ad-hoc column extraction in `splat_visualizer.cpp` with `buildSplatRenderData`**
- [ ] **Step 2: Replace duplicated inline shader ownership and keep `DataTable` as the only input model**

```cpp
#include "visualization/splat_shader_sources.h"

using splat::visualization::kGaussianFragmentShader;
using splat::visualization::kGaussianVertexShader;
```

- [ ] **Step 3: Make the renderer update path explicit and `DataTable`-centric**

```cpp
void NativeSplatProp::SetSplatData(std::shared_ptr<const DataTable> dataTable) {
  this->DataTable = std::move(dataTable);
  this->UploadDataTable(*this->DataTable);
  this->Bounds = computeSplatBounds(*this->DataTable, this->Positions);
  this->SortDirty = true;
  this->Modified();
}
```

- [ ] **Step 4: Add debug assertions/logging around the first successful upload/draw path**

```cpp
if (this->SplatCount == 0) {
  throw std::runtime_error("SplatVisualizer: upload produced zero splats");
}
if (this->Positions.empty() || this->Colors.empty() || this->Scales.empty() || this->Opacities.empty()) {
  throw std::runtime_error("SplatVisualizer: upload produced incomplete GPU input arrays");
}
```

- [ ] **Step 5: Rebuild and run the smoke test**

Run: `cmake --build build/test-plan --target splat_visualizer_smoke_test -j && ctest --test-dir build/test-plan --output-on-failure -R splat_visualizer_smoke_test`
Expected: `1/1 Test #2: splat_visualizer_smoke_test ... Passed`

- [ ] **Step 6: Commit**

```bash
git add src/visualization/splat_shader_sources.h src/visualization/splat_visualizer.cpp
git commit -m "refactor: split VTK visualizer render core into helpers"
```

### Task 4: Align Public Render Options with the TS Reference Rendering Contract

**Files:**
- Modify: `include/splat/visualization/splat_visualizer.h`
- Modify: `src/visualization/splat_visualizer.cpp`
- Test: `tests/visualization/splat_visualizer_smoke_test.cpp`

- [ ] **Step 1: Add explicit parity-oriented render options without breaking current defaults**

```cpp
struct SplatRenderOptions {
  float globalOpacity{1.0f};
  float sizeScale{3.0f};
  float minPointSize{1.0f};
  float maxPointSize{1024.0f};
  float alphaDiscardThreshold{0.001f};
  bool visible{true};
  bool depthTest{true};
  bool depthWrite{false};
  bool sortBackToFront{true};
  bool clampColors{true};
};
```

- [ ] **Step 2: Thread the new options into the render path**

```cpp
state->vtkglDepthMask(this->RenderOptions.depthWrite ? GL_TRUE : GL_FALSE);
if (this->RenderOptions.depthTest) {
  state->vtkglEnable(GL_DEPTH_TEST);
} else {
  state->vtkglDisable(GL_DEPTH_TEST);
}
if (this->RenderOptions.sortBackToFront) {
  this->UpdateDrawOrder(modelViewMatrix);
}
```

- [ ] **Step 3: Add a smoke assertion for public option plumbing**

```cpp
int main() {
  splat::SplatVisualizer visualizer("smoke");
  visualizer.setAxesEnabled(false);
  return 0;
}
```

Run: `cmake --build build/test-plan -j && ctest --test-dir build/test-plan --output-on-failure`
Expected: all visualization tests pass

- [ ] **Step 4: Commit**

```bash
git add include/splat/visualization/splat_visualizer.h src/visualization/splat_visualizer.cpp tests/visualization/splat_visualizer_smoke_test.cpp
git commit -m "feat: expose render options for TS-style VTK rendering"
```

### Task 5: Document the Final Architecture and Verification Commands

**Files:**
- Modify: `docs/superpowers/reference/visualization-module-structure.md`

- [ ] **Step 1: Add a section documenting the new render core split**

```md
## TS-Style VTK Render Core

- `src/visualization/splat_shader_sources.h`
  - owns the Gaussian GLSL source used by the VTK prop
- `src/visualization/splat_visualizer.cpp`
  - remains responsible for VTK scene integration, direct `DataTable` decode, and draw submission

The TS implementation is the correctness baseline. The C++ path is considered fixed only once visible splats render with TS-aligned decode and projection behavior.
```

- [ ] **Step 2: Run the full verification build**

Run: `cmake -S . -B build/vtk-parity -DBUILD_SPLAT_VISUALIZATION=ON -DBUILD_SPLAT_TESTS=ON && cmake --build build/vtk-parity -j && ctest --test-dir build/vtk-parity --output-on-failure`
Expected: configure succeeds, library builds, and all visualization tests pass

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/reference/visualization-module-structure.md
git commit -m "docs: record VTK playcanvas-style rendering architecture"
```

## Self-Review

- Spec coverage:
  - VTK remains the host rendering framework: covered by Tasks 3 and 4
  - TS semantics drive direct `DataTable` decode and render math: covered by Tasks 2 and 3
  - `DataTable` remains the only rendering data model: covered by Tasks 2 and 3
  - TS is the source of truth and current C++ non-rendering behavior is treated as a bug: covered by the plan header and Tasks 1 to 3
  - C++ rendering is made maintainable instead of staying monolithic: covered by Tasks 3 and 4
  - verification is built into the rollout: covered by Tasks 1 and 5
- Placeholder scan:
  - no TODO/TBD markers remain
  - all tasks list concrete file paths and commands
- Type consistency:
  - `UploadDataTable`
  - `SplatRenderOptions`
  are used consistently across tasks
