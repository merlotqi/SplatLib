#pragma once

#include <splat/models/data-table.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace splat::visualization {

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

struct GSplatData {
  std::vector<Vec3f> centers;
  std::vector<Vec4f> rotations;
  std::vector<Vec3f> scales;
  std::vector<Vec4f> colors;
  std::vector<std::uint32_t> sourceIndices;

  std::size_t size() const noexcept { return centers.size(); }
  bool empty() const noexcept { return centers.empty(); }
};

GSplatData adaptDataTableToGSplatData(const DataTable& dataTable, std::size_t maxSplats = 0);

float decodePlayCanvasColor(float dcValue);
float decodePlayCanvasOpacity(float opacityValue);
float decodePlayCanvasScale(float scaleValue);
Vec4f normalizePlayCanvasRotation(float w, float x, float y, float z);

}  // namespace splat::visualization