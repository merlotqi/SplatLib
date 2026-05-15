#pragma once

#include <splat/models/data-table.h>

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
  // PlayCanvas source layout: w, x, y, z, but stored in Vec4f fields as:
  //   x = w, y = x, z = y, w = z (intentional field mapping for PlayCanvas compatibility)
  std::vector<Vec4f> rotations;
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
