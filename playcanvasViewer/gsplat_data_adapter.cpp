#include "gsplat_data_adapter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace playcanvas_viewer {
namespace {
constexpr float kSHC0 = 0.28209479177387814f;
constexpr const char* kRequiredColumns[] = {"x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"};

void requireColumns(const splat::DataTable& dataTable) {
  for (const char* name : kRequiredColumns) {
    if (!dataTable.hasColumn(std::string(name))) {
      throw std::runtime_error(std::string("PlaycanvasViewer: missing required column: ") + name);
    }
  }
}

float columnValue(const splat::DataTable& dataTable, const char* name, size_t row) {
  return dataTable.getColumnByName(std::string(name)).getValue(row);
}
} // namespace

float decodePlayCanvasColor(float dcValue) {
  return 0.5f + dcValue * kSHC0;
}

float decodePlayCanvasOpacity(float v) {
  if (v > 0.0f) {
    return 1.0f / (1.0f + std::exp(-v));
  } else {
    float t = std::exp(v);
    return t / (1.0f + t);
  }
}

float decodePlayCanvasScale(float v) {
  return std::exp(v);
}

Vec4f normalizePlayCanvasRotation(float w, float x, float y, float z) {
  // If any input is non-finite (NaN or Inf), return identity quaternion.
  if (!(std::isfinite(w) && std::isfinite(x) && std::isfinite(y) && std::isfinite(z))) {
    return {1.0f, 0.0f, 0.0f, 0.0f};
  }
  float norm = std::sqrt(w*w + x*x + y*y + z*z);
  if (norm < std::numeric_limits<float>::epsilon()) {
    return {1.0f, 0.0f, 0.0f, 0.0f};
  }
  float iw = w / norm;
  float ix = x / norm;
  float iy = y / norm;
  float iz = z / norm;
  if (iw < 0.0f) {
    iw = -iw; ix = -ix; iy = -iy; iz = -iz;
  }
  return {iw, ix, iy, iz};
}

GSplatRenderData adaptDataTableToGSplat(const splat::DataTable& dataTable, size_t maxSplats) {
  requireColumns(dataTable);
  size_t rowCount = dataTable.getNumRows();
  size_t count = maxSplats == 0 ? rowCount : std::min(rowCount, maxSplats);
  GSplatRenderData out;
  out.centers.reserve(count);
  out.rotations.reserve(count);
  out.scales.reserve(count);
  out.colors.reserve(count);
  out.sourceIndices.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    out.centers.push_back({
      columnValue(dataTable, "x", i),
      columnValue(dataTable, "y", i),
      columnValue(dataTable, "z", i)
    });
    out.rotations.push_back(normalizePlayCanvasRotation(
      columnValue(dataTable, "rot_0", i),
      columnValue(dataTable, "rot_1", i),
      columnValue(dataTable, "rot_2", i),
      columnValue(dataTable, "rot_3", i)
    ));
    out.scales.push_back({
      decodePlayCanvasScale(columnValue(dataTable, "scale_0", i)),
      decodePlayCanvasScale(columnValue(dataTable, "scale_1", i)),
      decodePlayCanvasScale(columnValue(dataTable, "scale_2", i))
    });
    out.colors.push_back({
      decodePlayCanvasColor(columnValue(dataTable, "f_dc_0", i)),
      decodePlayCanvasColor(columnValue(dataTable, "f_dc_1", i)),
      decodePlayCanvasColor(columnValue(dataTable, "f_dc_2", i)),
      decodePlayCanvasOpacity(columnValue(dataTable, "opacity", i))
    });
    out.sourceIndices.push_back(static_cast<uint32_t>(i));
  }
  return out;
}

} // namespace playcanvas_viewer
