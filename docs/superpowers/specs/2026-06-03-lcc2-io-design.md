# LCC2 IO Design

## Context

LCC2 is XGRIDS' second-generation 3DGS container format. Unlike LCC1 (custom binary encoding per attribute), LCC2 delegates data storage to standard formats (PLY, SPZ, SOG). The LCC2 layer is purely a **spatial index + file manifest** wrapped in a JSON tree.

LCC1 IO was completed (`feat/xgrids-lcc` branch). SPZ v4 reader and writer were completed in the same branch. With PLY, SPZ, and SOG readers/writers all available, LCC2 becomes a thin orchestration layer.

**Reference**: `docs/LCC2Whitepaper/README.md` (Beta, v0.0.3)

---

## Architecture

```
                        scene.lcc2 (JSON tree)
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         data/3dgs/     data/mesh/     data/bvh/
          0.ply          0.ply          0.btree
          1.spz
          2.sog
```

### Reader Pipeline

```
scene.lcc2 → JSON parse → build Node tree
                              │
              遍历 leaf nodes → dispatch by extension
                              │
                    ┌─────────┼─────────┐
                  .ply       .spz      .sog
                    │          │         │
                 readPly   readSpz   readSog   (all return SplatCloud)
                    │          │         │
              按 name/start/count 切片 → 组装 Lcc2Scene
```

### Writer Pipeline

```
SplatCloud(s) → spatial partition (quadtree, same as LCC1)
                   │
              逐 cell 写入 data/3dgs/N.{ply|spz} (任意一种输出格式)
                   │
              构建 JSON tree (id, boundingBox, childNum, child, data.3dgs)
                   │
              写入 scene.lcc2
```

---

## Data Structures

### Lcc2Node (read-side)

```cpp
struct Lcc2NodeData3dgs {
  int name = -1;     // index into splatFiles[]
  int start = 0;     // first splat index in that file
  int count = 0;     // number of splats in this node
};

struct Lcc2NodeDataMesh {
  int name = -1;     // index into meshFiles[]
  int vertex = 0;
  int face = 0;
};

struct Lcc2Node {
  std::string id;                        // "0", "0-1", "0-1-7", ...
  Eigen::AlignedBox3f boundingBox;
  int childNum = 0;                      // 0 = leaf
  std::vector<std::unique_ptr<Lcc2Node>> children;

  // Leaf-only fields
  std::optional<Lcc2NodeData3dgs> d3dgs;
  std::optional<Lcc2NodeDataMesh> dmesh;
};
```

### Lcc2Scene (read-side product)

```cpp
struct Lcc2Scene {
  // Metadata
  std::string version;
  std::string name;
  std::string description;
  std::string guid;
  std::string fileType;          // "Quality" or "Portable"
  int totalSplats = 0;
  int totalLevels = 0;
  std::vector<int> lodSplats;    // per-LOD splat counts

  // File manifests
  std::vector<std::string> splatFiles;

  // Tree
  std::unique_ptr<Lcc2Node> root;

  // Loaded data (keyed by file index)
  std::map<int, SplatCloud> loadedData;

  // Coordinate system
  CoordinateSystem coordSystem = CoordinateSystem::RUB;
};
```

### Lcc2WriteConfig (write-side)

```cpp
struct Lcc2WriteConfig {
  float cellSizeX = 30.0f;
  float cellSizeY = 30.0f;
  std::string outputFormat = "ply";  // "ply" | "spz"
  std::string fileType = "Quality";  // "Quality" | "Portable"
  std::string name = "XGrids Splats";
  std::string description;
};
```

---

## Files to Create

| File | Purpose | Est. LOC |
|------|---------|----------|
| `include/splat/io/lcc2_reader.h` | Public API: `Lcc2Scene`, `readLcc2()` | 60 |
| `src/io/lcc2_reader.cpp` | JSON parse, tree build, format dispatch, slice | 250 |
| `include/splat/io/lcc2_writer.h` | Public API: `Lcc2WriteConfig`, `writeLcc2()` | 40 |
| `src/io/lcc2_writer.cpp` | Spatial partition, cell encode, tree JSON write | 200 |
| `tests/lcc2_test.cpp` | Round-trip test | 80 |

**Total**: ~630 lines. LCC1 was ~1200 lines. LCC2 is about half the work.

---

## Reader Implementation

### Step 1: Parse scene.lcc2 JSON

```cpp
Lcc2Scene readLcc2(const std::filesystem::path& lcc2Path) {
  std::ifstream f(lcc2Path);
  json j = json::parse(f);

  Lcc2Scene scene;
  scene.version    = j.value("version", "0.0.0");
  scene.name       = j.value("name", "");
  scene.fileType   = j.value("fileType", "Quality");
  scene.totalLevels = j.value("totalLevels", 1);

  // splatFiles
  for (auto& sf : j["splatFiles"])
    scene.splatFiles.push_back(sf.get<std::string>());

  // Build tree
  scene.root = parseNode(j["root"], scene);

  return scene;
}
```

### Step 2: Recursive node parser

```cpp
std::unique_ptr<Lcc2Node> parseNode(const json& j, Lcc2Scene& scene) {
  auto node = std::make_unique<Lcc2Node>();
  node->id = j["id"].get<std::string>();

  auto& bb = j["boundingBox"];
  node->boundingBox.min() = Eigen::Vector3f(bb["min"][0], bb["min"][1], bb["min"][2]);
  node->boundingBox.max() = Eigen::Vector3f(bb["max"][0], bb["max"][1], bb["max"][2]);

  node->childNum = j.value("childNum", 0);

  if (node->childNum > 0) {
    // Internal node: recurse into children
    for (auto& [key, childJ] : j["child"].items()) {
      node->children.push_back(parseNode(childJ, scene));
    }
  } else if (j.contains("data")) {
    // Leaf node: parse 3dgs/mesh/bvh references
    auto& data = j["data"];
    if (data.contains("3dgs")) {
      auto& d = data["3dgs"];
      node->d3dgs = Lcc2NodeData3dgs{
        d.value("name", -1),
        d.value("start", 0),
        d.value("count", 0)
      };
    }
    if (data.contains("mesh")) {
      auto& m = data["mesh"];
      node->dmesh = Lcc2NodeDataMesh{
        m.value("name", -1),
        m.value("vertex", 0),
        m.value("face", 0)
      };
    }
  }

  return node;
}
```

### Step 3: Lazy load + slice

```cpp
SplatCloud loadNodeData(Lcc2Scene& scene, const Lcc2Node& node,
                         const std::filesystem::path& baseDir) {
  if (!node.d3dgs || node.d3dgs->name < 0) return {};

  int fileIdx = node.d3dgs->name;
  const std::string& relPath = scene.splatFiles[fileIdx];

  // Lazy load: cache by file index
  if (!scene.loadedData.count(fileIdx)) {
    auto fullPath = baseDir / relPath;
    auto ext = fullPath.extension().string();
    if (ext == ".ply")      scene.loadedData[fileIdx] = readPly(fullPath);
    else if (ext == ".spz") scene.loadedData[fileIdx] = readSpz(fullPath);
    else if (ext == ".sog") scene.loadedData[fileIdx] = readSog(fullPath);
    else throw std::runtime_error("Unknown LCC2 data format: " + ext);
  }

  // Slice: copy [start, start+count) rows
  const auto& full = scene.loadedData[fileIdx];
  int start = node.d3dgs->start;
  int count = node.d3dgs->count;
  return full.slice(start, count);
}
```

### Step 4: SplatCloud::slice()

```cpp
// New method on SplatCloud
SplatCloud SplatCloud::slice(int start, int count) const {
  SplatCloud result;
  result.resize(count);
  result.setShDegree(shDegree());
  result.setCoordinateSystem(coordinateSystem());

  // Core columns: memcpy
  auto copyCol = [&](auto& dst, const auto& src) {
    std::memcpy(dst.data(), src.data() + start, count * sizeof(float));
  };
  copyCol(result.x(), x());
  copyCol(result.y(), y());
  // ... all 16 core columns
  // sh
  if (shDegree() > 0) {
    int shStride = shDim(shDegree()) * 3;
    std::memcpy(result.sh().data(), sh().data() + start * shStride,
                count * shStride * sizeof(float));
  }
  // extras
  for (auto& name : extraNames()) {
    result.addExtra(name);
    copyCol(result.extra(name), extra(name));
  }
  return result;
}
```

---

## Writer Implementation

### Step 1: Spatial partition (reuse from LCC1)

Same quadtree logic as `lcc_writer.cpp`:

```cpp
std::map<uint32_t, std::vector<size_t>> buildGrid(
    const SplatCloud& cloud, float cellSizeX, float cellSizeY) {
  std::map<uint32_t, std::vector<size_t>> grid;
  auto& xs = cloud.x(), &ys = cloud.y();
  Eigen::Vector3f bboxMin = computeBoundingBox(cloud).min();
  for (size_t i = 0; i < cloud.size(); ++i) {
    int cx = clamp(floor((xs[i] - bboxMin.x()) / cellSizeX), 0, 65535);
    int cy = clamp(floor((ys[i] - bboxMin.y()) / cellSizeY), 0, 65535);
    grid[(cy << 16) | cx].push_back(i);
  }
  return grid;
}
```

### Step 2: Write per-cell files

```cpp
void writeLcc2(const std::filesystem::path& outputDir,
               const std::vector<SplatCloud>& lods,
               const Lcc2WriteConfig& config) {

  auto dataDir = outputDir / "data" / "3dgs";
  fs::create_directories(dataDir);

  // For each cell, write a file
  int fileIdx = 0;
  std::vector<std::string> splatFiles;
  std::vector<Lcc2UnitWriteInfo> unitInfos;  // needed to build tree later

  for (auto& [cellId, indices] : grid) {
    auto cell = cloud.slice(indices);  // or selectRows

    std::string relPath = std::to_string(fileIdx) + "." + config.outputFormat;
    auto fullPath = dataDir / relPath;

    if (config.outputFormat == "ply") writePly(fullPath, cell);
    else if (config.outputFormat == "spz") writeSpz(fullPath, cell);
    else if (config.outputFormat == "sog") writeSog(fullPath, cell);

    splatFiles.push_back("data/3dgs/" + relPath);

    unitInfos.push_back({
      cellId,                 // (y<<16)|x
      fileIdx,                // name = index into splatFiles
      0,                      // start (always 0 for single-cell files)
      (int)cell.size(),       // count
      cell.boundingBox()
    });
    fileIdx++;
  }

  // Multi-LOD: same cellId appears across LODs; adjacent cells at same LOD
  // can be merged into the same file (per whitepaper: "adjacent Node data at
  // the same level will be merged and stored as the same file").
  // For simplicity, initial implementation: one file per cell per LOD.

  // Build tree from unitInfos
  auto root = buildTree(unitInfos, config);

  // Write scene.lcc2
  writeSceneLcc2(outputDir / (config.name + ".lcc2"), root, splatFiles, config);
}
```

### Step 3: Build tree JSON

```cpp
json nodeToJson(const Lcc2NodeWrite& node) {
  json j;
  j["id"] = node.id;
  j["boundingBox"] = {
    {"min", {node.bbox.min().x(), node.bbox.min().y(), node.bbox.min().z()}},
    {"max", {node.bbox.max().x(), node.bbox.max().y(), node.bbox.max().z()}}
  };
  j["childNum"] = node.children.size();

  if (node.children.empty()) {
    // Leaf
    if (node.hasData) {
      j["data"] = {
        {"3dgs", {{"name", node.fileIdx}, {"start", node.start}, {"count", node.count}}}
      };
    }
  } else {
    // Internal
    json childObj;
    for (size_t i = 0; i < node.children.size(); ++i)
      childObj[std::to_string(i)] = nodeToJson(*node.children[i]);
    j["child"] = childObj;
  }

  return j;
}
```

---

## Public API

```cpp
// include/splat/io/lcc2_reader.h

struct Lcc2Scene;
struct Lcc2Node;

Lcc2Scene readLcc2(const std::filesystem::path& lcc2Path);

// Utility: collect all leaf splats into one flat cloud
SplatCloud flattenLcc2Scene(const Lcc2Scene& scene);

// include/splat/io/lcc2_writer.h

struct Lcc2WriteConfig;

void writeLcc2(const std::filesystem::path& outputDir,
               const std::vector<SplatCloud>& lods,
               const Lcc2WriteConfig& config = {});
```

---

## Verification

1. **JSON round-trip**: build tree → serialize JSON → parse → tree matches
2. **SplatCloud slice**: create cloud with known values → slice(5, 3) → verify 3 rows match
3. **Format dispatch**: create temp files of each type → `readLcc2` dispatches correctly
4. **Full round-trip**: `SplatCloud → writeLcc2 → readLcc2 → flattenLcc2Scene → SplatCloud` matches within quantization tolerance
5. **Multi-LOD**: 3 LOD levels → write → read → verify per-LOD node counts match

---

## What This Design Does NOT Cover

- **Mesh collision data**: `data/mesh/` files exist but `Lcc2NodeDataMesh` is parsed-only. Mesh-BVH intersection is a separate feature.
- **`.btree` BVH**: Format undefined by whitepaper. Skipped.
- **Multi-file merging**: Adjacent same-level nodes "merged and stored as the same file" — initial implementation uses one-file-per-cell for simplicity. Optimize later.
- **`virtualLoD`**: Parsed from JSON but not implemented (runtime LOD selection is a renderer concern).
- **`renderingHints` / `splatExtraAttributes` / `env`**: Parsed and round-tripped through JSON, no semantic processing.
