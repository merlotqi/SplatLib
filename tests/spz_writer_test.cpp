/**
 * @file spz_writer_test.cpp
 * @brief Round-trip test: SplatCloud -> writeSpz -> readSpz -> verify
 */
#include <splat/io/spz_reader.h>
#include <splat/io/spz_writer.h>
#include <splat/models/splatcloud.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

static void check(bool cond, const char* msg) {
  if (!cond) {
    fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
  }
}

static splat::SplatCloud makeTestTable(int n, bool withSH) {
  std::vector<splat::Column> cols;
  auto addCol = [&](const char* name) -> float* {
    std::vector<float> d(n);
    float* ptr = d.data();
    cols.push_back({name, std::move(d)});
    return ptr;
  };

  float *xs = addCol("x"), *ys = addCol("y"), *zs = addCol("z");
  float *s0 = addCol("scale_0"), *s1 = addCol("scale_1"), *s2 = addCol("scale_2");
  float *dc0 = addCol("f_dc_0"), *dc1 = addCol("f_dc_1"), *dc2 = addCol("f_dc_2");
  float* op = addCol("opacity");
  float *r0 = addCol("rot_0"), *r1 = addCol("rot_1"), *r2 = addCol("rot_2"), *r3 = addCol("rot_3");

  for (int i = 0; i < n; ++i) {
    xs[i] = i * 1.0f;
    ys[i] = i * 2.0f;
    zs[i] = i * 3.0f;
    s0[i] = -1.0f;
    s1[i] = 0.0f;
    s2[i] = 1.0f;
    dc0[i] = 0.1f;
    dc1[i] = -0.1f;
    dc2[i] = 0.2f;
    op[i] = 0.5f;
    r0[i] = 1.0f;
    r1[i] = 0.0f;
    r2[i] = 0.0f;
    r3[i] = 0.0f;
  }

  if (withSH) {
    for (int h = 0; h < 9; ++h) {
      std::vector<float> d(n, (h - 4.0f) * 0.2f);
      cols.push_back({"f_rest_" + std::to_string(h), std::move(d)});
    }
  }

  return splat::SplatCloud(std::move(cols));
}

int main() {
  auto tmp = std::filesystem::temp_directory_path() / "test_roundtrip.spz";

  // Test 1: No SH
  {
    auto dt = makeTestTable(10, false);
    splat::writeSpz(tmp, dt);
    auto result = splat::readSpz(tmp);
    check(result->getNumRows() == 10, "round-trip no-SH: row count");
    check(!result->hasColumn("f_rest_0"), "round-trip no-SH: no SH column");
    auto& xCol = result->getColumnByName("x");
    check(std::fabs(xCol.getValue<float>(0) - 0.0f) < 0.001f, "round-trip no-SH: x[0]");
    printf("PASS: round-trip no SH\n");
  }

  // Test 2: With SH
  {
    auto dt = makeTestTable(3, true);
    splat::writeSpz(tmp, dt);
    auto result = splat::readSpz(tmp);
    check(result->getNumRows() == 3, "round-trip SH: row count");
    check(result->hasColumn("f_rest_0"), "round-trip SH: has SH column");
    printf("PASS: round-trip with SH\n");
  }

  // Test 3: Single splat
  {
    auto dt = makeTestTable(1, false);
    splat::writeSpz(tmp, dt);
    auto result = splat::readSpz(tmp);
    check(result->getNumRows() == 1, "single splat: row count");
    printf("PASS: single splat\n");
  }

  std::filesystem::remove(tmp);
  printf("\n=== All SPZ writer tests passed ===\n");
  return 0;
}
