#include "gsplat_data_adapter.h"

namespace playcanvas_viewer {

float decodePlayCanvasColor(float dcValue) { return splat::visualization::decodePlayCanvasColor(dcValue); }

float decodePlayCanvasOpacity(float opacityValue) {
  return splat::visualization::decodePlayCanvasOpacity(opacityValue);
}

float decodePlayCanvasScale(float scaleValue) { return splat::visualization::decodePlayCanvasScale(scaleValue); }

Vec4f normalizePlayCanvasRotation(float w, float x, float y, float z) {
  return splat::visualization::normalizePlayCanvasRotation(w, x, y, z);
}

GSplatRenderData adaptDataTableToGSplat(const splat::SplatCloud& dataTable, size_t maxSplats) {
  return splat::visualization::adaptDataTableToGSplatData(dataTable, maxSplats);
}

}  // namespace playcanvas_viewer
