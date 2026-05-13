# Transform And Playcanvas LOD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Translate the pinned TypeScript CLI behavior needed for LOD generation into `transform/`, and add an independent `playcanvasLOD/` executable that generates PlayCanvas-ready multi-level LOD output from one input file.

**Architecture:** `transform/` remains the general CLI and gains a real file-action parser plus an executable process pipeline for `param`, `lod`, and `decimate`. `playcanvasLOD/` is a focused tool that links only `SPLAT::splat`, reads one source file through public readers, generates simplified copies from the original input for each requested percentage, assigns `lod` values in argument order, combines the results, and writes `lod-meta.json`.

**Tech Stack:** C++17, CMake, Abseil flags/strings, Eigen, public `splat` IO and op APIs, manual CLI verification commands

---

### Task 1: Extend `transform` action model for decimation

**Files:**
- Modify: `transform/process.h`
- Modify: `transform/process.cpp`
- Test: manual build and CLI validation in later tasks

- [ ] **Step 1: Add the `Decimate` action type to `transform/process.h`**

```cpp
struct Decimate {
  int count = -1;
  float percent = -1.0f;
};

using ProcessAction = std::variant<
    Translate,
    Rotate,
    Scale,
    FilterNaN,
    FilterByValue,
    FilterBands,
    FilterBox,
    FilterSphere,
    Param,
    Lod,
    Decimate>;
```

- [ ] **Step 2: Include the public APIs needed by the processor**

Add these includes near the top of `transform/process.cpp`:

```cpp
#include <splat/op/decimate.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>
```

- [ ] **Step 3: Add small helpers for `lod` and `decimate` execution**

Insert helper functions above `processDataTable()`:

```cpp
static Column* findColumn(DataTable* dataTable, const std::string& name) {
  for (auto& column : dataTable->columns) {
    if (column.name == name) {
      return &column;
    }
  }
  return nullptr;
}

static void ensureLodColumn(DataTable* dataTable) {
  if (!findColumn(dataTable, "lod")) {
    dataTable->addColumn({"lod", std::vector<float>(dataTable->getNumRows())});
  }
}

static int computeKeepCount(size_t numRows, const Decimate& action) {
  if (action.count >= 0) {
    return std::min<int>(action.count, static_cast<int>(numRows));
  }
  if (action.percent >= 0.0f) {
    const float raw = static_cast<float>(numRows) * action.percent / 100.0f;
    return std::clamp<int>(static_cast<int>(std::lround(raw)), 0, static_cast<int>(numRows));
  }
  throw std::runtime_error("Decimate action missing count and percent");
}
```

- [ ] **Step 4: Implement the first working `processDataTable()` pipeline**

Replace the empty loop body with explicit action handling for the first required subset:

```cpp
std::unique_ptr<DataTable> processDataTable(DataTable* dataTable, const std::vector<ProcessAction>& processActions) {
  assert(dataTable);
  std::unique_ptr<DataTable> result(dataTable);

  for (const auto& action : processActions) {
    std::visit(
        [&](const auto& typedAction) {
          using T = std::decay_t<decltype(typedAction)>;
          if constexpr (std::is_same_v<T, Param>) {
            return;
          } else if constexpr (std::is_same_v<T, Lod>) {
            ensureLodColumn(result.get());
            findColumn(result.get(), "lod")->asVector<float>().assign(result->getNumRows(),
                                                                      static_cast<float>(typedAction.value));
          } else if constexpr (std::is_same_v<T, Decimate>) {
            const int keepCount = computeKeepCount(result->getNumRows(), typedAction);
            result = simplifyGaussians(*result, keepCount);
          } else {
            throw std::runtime_error("Unsupported process action in current translation");
          }
        },
        action);
  }

  return result;
}
```

- [ ] **Step 5: Run a compile-only check for the updated processor**

Run:

```bash
cmake -S . -B build/plan-check -DBUILD_SPLAT_TRANSFORM_TOOL=ON -DBUILD_SPLAT_PLAYCANVAS_LOD=OFF
cmake --build build/plan-check --target SplatTransform -j
```

Expected: `SplatTransform` builds successfully.

- [ ] **Step 6: Commit the action model and processor**

```bash
git add transform/process.h transform/process.cpp
git commit -m "feat: add transform decimate action pipeline"
```

### Task 2: Translate the required `transform` CLI file-action parser

**Files:**
- Modify: `transform/main.cpp`
- Test: manual CLI validation in this task and Task 5

- [ ] **Step 1: Add parsing helpers for counts, percentages, and key-value params**

Add helper structs and parsing functions above `parseArguments()`:

```cpp
struct ParsedDecimate {
  int count = -1;
  float percent = -1.0f;
};

static ParsedDecimate parseDecimateValue(absl::string_view value) {
  ParsedDecimate result;
  std::string text(value);
  if (!text.empty() && text.back() == '%') {
    text.pop_back();
    float percent = 0.0f;
    if (!absl::SimpleAtof(text, &percent) || percent < 0.0f || percent > 100.0f) {
      throw std::runtime_error("Invalid decimate percentage: " + std::string(value));
    }
    result.percent = percent;
    return result;
  }

  int count = 0;
  if (!absl::SimpleAtoi(value, &count) || count < 0) {
    throw std::runtime_error("Invalid decimate count: " + std::string(value));
  }
  result.count = count;
  return result;
}

static std::vector<Param> parseParamsValue(absl::string_view value) {
  std::vector<Param> params;
  for (auto item : absl::StrSplit(value, ',', absl::SkipEmpty())) {
    std::vector<absl::string_view> parts = absl::StrSplit(item, '=');
    params.push_back({std::string(parts[0]), parts.size() > 1 ? std::string(parts[1]) : ""});
  }
  return params;
}
```

- [ ] **Step 2: Declare the missing repeated file-action flags**

Add these flag declarations beside the existing `--lod` flag:

```cpp
ABSL_FLAG(std::vector<std::string>, decimate, {}, "Simplify to n or n% Gaussians");
ABSL_FLAG(std::vector<std::string>, params, {}, "Additional parameters in key=value form");
```

- [ ] **Step 3: Replace the current placeholder file parsing with explicit argv scanning**

Inside `parseArguments()`, iterate over `argv[1..argc)` and attach actions to the most recent file:

```cpp
for (int i = 1; i < argc; ++i) {
  const std::string arg = argv[i];
  if (arg == "--") {
    continue;
  }
  if (!arg.empty() && arg[0] != '-') {
    files.push_back({std::filesystem::u8path(arg), {}});
    continue;
  }
  if (files.empty()) {
    continue;
  }

  auto requireValue = [&](const std::string& optionName) -> std::string {
    if (i + 1 >= argc) {
      throw std::runtime_error("Missing value for option " + optionName);
    }
    ++i;
    return argv[i];
  };

  File& current = files.back();
  if (arg == "--lod") {
    const int lod = parseInteger(requireValue(arg));
    if (lod < 0) {
      throw std::runtime_error("Invalid lod value");
    }
    current.processActions.push_back(Lod{lod});
  } else if (arg == "--decimate") {
    const auto parsed = parseDecimateValue(requireValue(arg));
    current.processActions.push_back(Decimate{parsed.count, parsed.percent});
  } else if (arg == "--params") {
    for (const auto& param : parseParamsValue(requireValue(arg))) {
      current.processActions.push_back(param);
    }
  }
}
```

- [ ] **Step 4: Add basic structural validation before returning parsed files**

Append these checks at the end of `parseArguments()`:

```cpp
if (files.size() < 2) {
  throw std::runtime_error("Expected at least one input file and one output file");
}
```

- [ ] **Step 5: Update help text to document the new translated subset**

Adjust the file-actions section in `main()` to include:

```cpp
std::cout << "  --decimate <n|n%>             Simplify to n (or n%) Gaussians\n";
std::cout << "  --lod <n>                     Specify the level of detail, n >= 0\n";
std::cout << "  --params <key=value,...>      Additional parameters\n";
```

- [ ] **Step 6: Build and smoke-test `transform` argument parsing**

Run:

```bash
cmake --build build/plan-check --target SplatTransform -j
./build/plan-check/transform/SplatTransform --help
```

Expected: the binary builds and the help text lists `--decimate`, `--lod`, and `--params`.

- [ ] **Step 7: Commit the parser translation**

```bash
git add transform/main.cpp
git commit -m "feat: translate transform LOD action parsing"
```

### Task 3: Add the independent `playcanvasLOD` executable

**Files:**
- Add: `playcanvasLOD/CMakeLists.txt`
- Add: `playcanvasLOD/main.cpp`
- Modify: `CMakeLists.txt`
- Test: build and end-to-end command in later steps

- [ ] **Step 1: Add a top-level build option for the new tool**

Insert into `CMakeLists.txt` near the existing tool options:

```cmake
option(BUILD_SPLAT_PLAYCANVAS_LOD "Build PlayCanvas LOD generation tool" OFF)
```

And add:

```cmake
if(BUILD_SPLAT_PLAYCANVAS_LOD)
    add_subdirectory(playcanvasLOD)
endif()
```

- [ ] **Step 2: Create `playcanvasLOD/CMakeLists.txt`**

Use a minimal executable definition that links only public library targets plus CLI dependencies:

```cmake
if(UNIX)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ABSL_FLAGS REQUIRED absl_flags_parse absl_flags_usage absl_flags_marshalling absl_flags_program_name absl_flags_reflection absl_flags_commandlineflag)
endif()

add_executable(PlaycanvasLOD main.cpp)
target_link_libraries(PlaycanvasLOD PRIVATE SPLAT::splat)

if(UNIX)
    target_link_libraries(PlaycanvasLOD PRIVATE
        absl_flags_parse
        absl_flags_usage
        absl_flags_reflection
        absl_flags_commandlineflag
        absl_flags_marshalling
        absl_flags_internal
        absl_flags_program_name
        absl_flags_config
        absl_raw_logging_internal
    )
elseif(WIN32)
    target_link_libraries(PlaycanvasLOD PRIVATE
        absl::flags_parse
        absl::flags
    )
endif()
```

- [ ] **Step 3: Implement the minimal `playcanvasLOD/main.cpp` CLI**

Create a single-file tool with:

```cpp
ABSL_FLAG(bool, overwrite, false, "Overwrite output");
ABSL_FLAG(int32_t, iterations, 10, "Iterations for SOG SH compression");
ABSL_FLAG(int32_t, lod_chunk_count, 64, "Approximate number of Gaussians per LOD chunk in K");
ABSL_FLAG(int32_t, lod_chunk_extent, 16, "Approximate size of an LOD chunk in world units (m)");
ABSL_FLAG(std::vector<std::string>, levels, {}, "LOD percentages such as 100% 50% 25%");
```

Required implementation blocks:

```cpp
static std::unique_ptr<DataTable> readSingleInput(const std::filesystem::path& filename);
static std::vector<float> parseLevels(const std::vector<std::string>& values);
static void ensureLodColumn(DataTable* dataTable);
static bool isGSDataTable(const DataTable* dataTable);
```

Main flow:

```cpp
auto args = absl::ParseCommandLine(argc, argv);
if (args.size() != 3) { /* print usage and fail */ }

const auto input = std::filesystem::absolute(std::filesystem::u8path(args[1]));
const auto output = std::filesystem::absolute(std::filesystem::u8path(args[2]));
auto levels = parseLevels(absl::GetFlag(FLAGS_levels));
auto source = readSingleInput(input);

if (!source || source->getNumRows() == 0 || !isGSDataTable(source.get())) {
  throw std::runtime_error("Unsupported Gaussian input");
}

std::vector<std::unique_ptr<DataTable>> generated;
for (size_t level = 0; level < levels.size(); ++level) {
  const int keep = std::clamp<int>(
      static_cast<int>(std::lround(source->getNumRows() * levels[level] / 100.0f)),
      0,
      static_cast<int>(source->getNumRows()));
  auto simplified = simplifyGaussians(*source, keep);
  ensureLodColumn(simplified.get());
  simplified->getColumnByName("lod").asVector<float>().assign(
      simplified->getNumRows(), static_cast<float>(level));
  generated.emplace_back(std::move(simplified));
}

auto merged = combine(generated);
writeLod(output, merged.get(), nullptr, true, absl::GetFlag(FLAGS_iterations),
         absl::GetFlag(FLAGS_lod_chunk_count), absl::GetFlag(FLAGS_lod_chunk_extent));
```

- [ ] **Step 4: Implement the public-reader dispatcher in `playcanvasLOD/main.cpp`**

Support the formats already reachable from public headers:

```cpp
if (absl::EndsWithIgnoreCase(u8, ".ksplat")) return readKsplat(filename);
if (absl::EndsWithIgnoreCase(u8, ".splat")) return readSplat(filename);
if (absl::EndsWithIgnoreCase(u8, ".sog") || absl::EndsWithIgnoreCase(u8, "meta.json")) return readSog(filename, filename);
if (absl::EndsWithIgnoreCase(u8, ".ply")) return readPly(filename);
if (absl::EndsWithIgnoreCase(u8, ".spz")) return readSpz(filename);
if (absl::EndsWithIgnoreCase(u8, ".voxel.json")) return readVoxel(filename);
throw std::runtime_error("Unsupported input file type: " + u8);
```

- [ ] **Step 5: Build the new executable**

Run:

```bash
cmake -S . -B build/plan-check -DBUILD_SPLAT_TRANSFORM_TOOL=ON -DBUILD_SPLAT_PLAYCANVAS_LOD=ON
cmake --build build/plan-check --target PlaycanvasLOD -j
```

Expected: `PlaycanvasLOD` builds successfully.

- [ ] **Step 6: Commit the new tool scaffold**

```bash
git add CMakeLists.txt playcanvasLOD/CMakeLists.txt playcanvasLOD/main.cpp
git commit -m "feat: add playcanvas LOD generator tool"
```

### Task 4: Align `transform` output flow with the translated LOD generation behavior

**Files:**
- Modify: `transform/main.cpp`
- Test: end-to-end `transform` command in this task

- [ ] **Step 1: Make output-side actions run on the combined non-environment table**

Keep or adjust the existing combine flow so this line remains the decisive post-combine action stage:

```cpp
dataTable.reset(processDataTable(combine(nonEnvDataTables).release(), outputArg.processActions).release());
```

The key requirement is that output-side `--decimate` and `--lod` act on the merged table, matching the current spec.

- [ ] **Step 2: Preserve environment table handling unchanged**

Do not regress this branch:

```cpp
if (!envDataTables.empty()) {
  envDataTable = processDataTable(combine(envDataTables).release(), outputArg.processActions);
}
```

If output-side `Decimate` should never affect environment data in practice, guard it inside `processDataTable()` or skip the action for tables whose `lod` column is entirely `-1`.

- [ ] **Step 3: Run an end-to-end `transform` LOD smoke test**

Use a real sample file available in the repo or local fixture and run:

```bash
./build/plan-check/transform/SplatTransform \
  input.ply \
  out/lod-meta.json \
  --decimate 100% \
  --lod 0
```

If the parser requires file-local options before the output positional argument, use:

```bash
./build/plan-check/transform/SplatTransform \
  input.ply \
  out/lod-meta.json --decimate 100% --lod 0
```

Expected: output directory is created and contains `lod-meta.json`.

- [ ] **Step 4: Commit the stabilized transform flow**

```bash
git add transform/main.cpp
git commit -m "fix: wire transform output LOD flow"
```

### Task 5: Verify both binaries with real commands and capture any format limits

**Files:**
- Modify: `docs/superpowers/specs/2026-05-13-transform-playcanvas-lod-design.md` only if supported input formats or constraints need clarification after implementation
- Test: `build/plan-check` binaries

- [ ] **Step 1: Run the final combined build**

```bash
cmake -S . -B build/plan-check -DBUILD_SPLAT_TRANSFORM_TOOL=ON -DBUILD_SPLAT_PLAYCANVAS_LOD=ON
cmake --build build/plan-check -j
```

Expected: both `SplatTransform` and `PlaycanvasLOD` build successfully.

- [ ] **Step 2: Run an end-to-end `PlaycanvasLOD` command**

```bash
./build/plan-check/playcanvasLOD/PlaycanvasLOD \
  input.ply \
  out/playcanvas/lod-meta.json \
  --levels=100% --levels=50% --levels=25%
```

Expected: output tree contains `lod-meta.json` and generated LOD chunk files.

- [ ] **Step 3: Inspect that generated `lod` values match level order**

Use a quick inspection command or a small debug read to verify:

```bash
find out/playcanvas -maxdepth 2 -type f | sort
```

Expected: files exist for LOD groups corresponding to levels `0`, `1`, and `2`.

- [ ] **Step 4: If needed, document final supported inputs or known constraints**

If implementation reveals a narrower format set than expected, append a short clarification to:

```text
docs/superpowers/specs/2026-05-13-transform-playcanvas-lod-design.md
```

- [ ] **Step 5: Commit verification-driven follow-ups**

```bash
git add docs/superpowers/specs/2026-05-13-transform-playcanvas-lod-design.md
git commit -m "docs: clarify implemented LOD tool constraints"
```

Skip this commit if no doc change is needed.
