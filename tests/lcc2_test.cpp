#include <splat/io/lcc2_reader.h>
#include <splat/io/lcc2_writer.h>
#include <splat/models/splatcloud.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

static void check(bool cond, const char* msg) {
  if (!cond) {
    fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
  }
}

static splat::SplatCloud makeTable(int n) {
  std::vector<splat::Column> cols;
  auto add = [&](const char* name) -> float* {
    std::vector<float> d(n);
    float* p = d.data();
    cols.push_back({name, std::move(d)});
    return p;
  };
  float *xs = add("x"), *ys = add("y"), *zs = add("z");
  float *s0 = add("scale_0"), *s1 = add("scale_1"), *s2 = add("scale_2");
  float *cr = add("f_dc_0"), *cg = add("f_dc_1"), *cb = add("f_dc_2");
  float* op = add("opacity");
  float *r0 = add("rot_0"), *r1 = add("rot_1"), *r2 = add("rot_2"), *r3 = add("rot_3");

  for (int i = 0; i < n; ++i) {
    xs[i] = i * 1.0f;
    ys[i] = i * 2.0f;
    zs[i] = i * 3.0f;
    s0[i] = -1;
    s1[i] = 0;
    s2[i] = 1;
    cr[i] = 0.1f;
    cg[i] = -0.1f;
    cb[i] = 0.2f;
    op[i] = 0.5f;
    r0[i] = 1;
    r1[i] = 0;
    r2[i] = 0;
    r3[i] = 0;
  }
  return splat::SplatCloud(std::move(cols));
}

int main() {
  namespace fs = std::filesystem;
  using namespace splat;

  // ── Test 1: Reader parses synthetic JSON ──
  {
    auto tmpDir = fs::temp_directory_path() / "lcc2_test_read";
    fs::create_directories(tmpDir);
    auto lcc2Path = tmpDir / "test.lcc2";

    {
      std::ofstream f(lcc2Path);
      f << R"({
        "version": "0.0.3",
        "name": "TestScene",
        "fileType": "Quality",
        "totalSplats": 100,
        "totalLevels": 1,
        "lodSplats": [100],
        "splatFiles": [],
        "root": {
          "id": "0",
          "boundingBox": {"min": [0,0,0], "max": [10,10,10]},
          "childNum": 1,
          "child": {
            "0": {
              "id": "0-0",
              "boundingBox": {"min": [0,0,0], "max": [5,5,5]},
              "childNum": 0,
              "data": {"3dgs": {"name": -1, "start": 0, "count": 0}}
            }
          }
        }
      })";
    }

    auto scene = readLcc2(lcc2Path);
    check(scene.version == "0.0.3", "version");
    check(scene.name == "TestScene", "name");
    check(scene.fileType == "Quality", "fileType");
    check(scene.totalSplats == 100, "totalSplats");
    check(scene.root != nullptr, "root exists");
    check(scene.root->children.size() == 1, "root has 1 child");
    check(scene.root->children[0]->isLeaf(), "child is leaf");
    check(scene.root->children[0]->d3dgs.has_value(), "child has 3dgs");
    printf("PASS: reader JSON parse\n");
    fs::remove_all(tmpDir);
  }

  // ── Test 2: Writer round-trip ──
  {
    auto tmpDir = fs::temp_directory_path() / "lcc2_test_rt";
    fs::create_directories(tmpDir);

    auto dt = makeTable(5);
    Lcc2WriteConfig cfg;
    cfg.name = "roundtrip";
    cfg.outputFormat = "spz";
    cfg.cellSizeX = 100.0f;
    cfg.cellSizeY = 100.0f;

    writeLcc2(tmpDir, {&dt}, cfg);
    check(fs::exists(tmpDir / "roundtrip.lcc2"), "scene file created");
    check(fs::exists(tmpDir / "data" / "3dgs"), "data dir created");
    printf("PASS: writer output files\n");

    auto scene = readLcc2(tmpDir / "roundtrip.lcc2");
    check(scene.totalSplats == 5, "totalSplats preserved");
    check(scene.totalLevels == 1, "totalLevels preserved");
    check(!scene.splatFiles.empty(), "splatFiles non-empty");
    printf("PASS: writer round-trip\n");

    fs::remove_all(tmpDir);
  }

  printf("\n=== All LCC2 tests passed ===\n");
  return 0;
}
