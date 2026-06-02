/**
 * @file coordinate_converter_test.cpp
 * @brief Unit tests for CoordinateConverter.
 */
#include <splat/maths/coordinate-converter.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

static void check(bool cond, const char* msg) {
  if (!cond) { fprintf(stderr, "FAIL: %s\n", msg); std::exit(1); }
}

int main() {
  using CS = splat::CoordinateSystem;

  // Test 1: identity (RUB -> RUB)
  {
    auto conv = splat::makeCoordinateConverter(CS::RUB, CS::RUB, 0);
    float p[3] = {1, 2, 3};
    conv.convertPosition(p);
    check(p[0] == 1 && p[1] == 2 && p[2] == 3, "identity position");
    printf("PASS: identity RUB->RUB\n");
  }

  // Test 2: within-family (RUB -> RUF, z flip)
  {
    auto conv = splat::makeCoordinateConverter(CS::RUB, CS::RUF, 0);
    float p[3] = {1, 2, 3};
    conv.convertPosition(p);
    check(p[0] == 1 && p[1] == 2 && p[2] == -3, "RUB->RUF z-flip");
    printf("PASS: within-family RUB->RUF\n");
  }

  // Test 3: cross-family position round-trip (RUB -> RDF -> RUB)
  {
    auto toRDF = splat::makeCoordinateConverter(CS::RUB, CS::RDF, 0);
    auto toRUB = splat::makeCoordinateConverter(CS::RDF, CS::RUB, 0);
    float p[3] = {1, 2, 3};
    toRDF.convertPosition(p);
    toRUB.convertPosition(p);
    check(std::fabs(p[0] - 1) < 0.001f && std::fabs(p[1] - 2) < 0.001f &&
          std::fabs(p[2] - 3) < 0.001f, "cross-family position round-trip");
    printf("PASS: cross-family position RUB<->RDF round-trip\n");
  }

  // Test 4: cross-family quaternion round-trip
  {
    auto toRDF = splat::makeCoordinateConverter(CS::RUB, CS::RDF, 1);
    auto toRUB = splat::makeCoordinateConverter(CS::RDF, CS::RUB, 1);
    float q[4] = {1, 0, 0, 0};  // identity quat
    toRDF.convertRotation(q);
    toRUB.convertRotation(q);
    check(std::fabs(q[0] - 1) < 0.001f && std::fabs(q[1]) < 0.001f &&
          std::fabs(q[2]) < 0.001f && std::fabs(q[3]) < 0.001f,
          "cross-family quaternion round-trip");
    printf("PASS: cross-family quaternion round-trip\n");
  }

  // Test 5: all 16 systems convert without crash
  for (int from = 1; from <= 16; ++from) {
    auto conv = splat::makeCoordinateConverter(static_cast<CS>(from), CS::RUB, 0);
    float p[3] = {1, 2, 3};
    conv.convertPosition(p);
  }
  printf("PASS: all 16 coordinate systems\n");

  printf("\n=== All coordinate converter tests passed ===\n");
  return 0;
}
