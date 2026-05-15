#pragma once

#include <splat/models/data-table.h>
#include <splat/visualization/gsplat_data.h>

#include <cstddef>

namespace playcanvas_viewer {

using Vec3f = splat::visualization::Vec3f;
using Vec4f = splat::visualization::Vec4f;
using GSplatRenderData = splat::visualization::GSplatData;

GSplatRenderData adaptDataTableToGSplat(const splat::DataTable& dataTable, size_t maxSplats = 0);

float decodePlayCanvasColor(float dcValue);
float decodePlayCanvasOpacity(float opacityValue);
float decodePlayCanvasScale(float scaleValue);
Vec4f normalizePlayCanvasRotation(float w, float x, float y, float z);

}  // namespace playcanvas_viewer
