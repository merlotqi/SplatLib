/***********************************************************************************
 *
 * splat - A C++ library for reading and writing 3D Gaussian Splatting (splat) files.
 *
 * This library provides functionality to convert, manipulate, and process
 * 3D Gaussian splatting data formats used in real-time neural rendering.
 *
 * This file is part of splat.
 *
 * splat is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * splat is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * For more information, visit the project's homepage or contact the author.
 *
 ***********************************************************************************/

#include <splat/io/splat-writer.h>
#include <splat/maths/maths.h>
#include <splat/models/data-table.h>

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace splat {

static uint8_t float2UInt8(float v) {
  return static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, std::round(v * 255.0f))));
}

void writeSplat(const DataTable* datatable, const std::string& filepath) {
  std::ofstream ofs(filepath, std::ios::binary | std::ios::out);
  if (!ofs.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filepath);
  }

  auto&& numSplats = datatable->getNumRows();

  auto&& col_x = datatable->getColumnByName("x").asSpan<float>();
  auto&& col_y = datatable->getColumnByName("y").asSpan<float>();
  auto&& col_z = datatable->getColumnByName("z").asSpan<float>();

  auto&& col_scale_0 = datatable->getColumnByName("scale_0").asSpan<float>();
  auto&& col_scale_1 = datatable->getColumnByName("scale_1").asSpan<float>();
  auto&& col_scale_2 = datatable->getColumnByName("scale_2").asSpan<float>();

  auto&& col_f_dc_0 = datatable->getColumnByName("f_dc_0").asSpan<float>();
  auto&& col_f_dc_1 = datatable->getColumnByName("f_dc_1").asSpan<float>();
  auto&& col_f_dc_2 = datatable->getColumnByName("f_dc_2").asSpan<float>();
  auto&& col_opacity = datatable->getColumnByName("opacity").asSpan<float>();

  auto&& col_rot_0 = datatable->getColumnByName("rot_0").asSpan<float>();
  auto&& col_rot_1 = datatable->getColumnByName("rot_1").asSpan<float>();
  auto&& col_rot_2 = datatable->getColumnByName("rot_2").asSpan<float>();
  auto&& col_rot_3 = datatable->getColumnByName("rot_3").asSpan<float>();

  std::vector<uint8_t> buffer(32);
  static constexpr float SH_C0 = 0.28209479177387814f;

  for (size_t i = 0; i < numSplats; ++i) {
    std::memcpy(buffer.data() + 0, &col_x[i], 4);
    std::memcpy(buffer.data() + 4, &col_y[i], 4);
    std::memcpy(buffer.data() + 8, &col_z[i], 4);

    float s0 = std::exp(col_scale_0[i]);
    float s1 = std::exp(col_scale_1[i]);
    float s2 = std::exp(col_scale_2[i]);
    std::memcpy(buffer.data() + 12, &s0, 4);
    std::memcpy(buffer.data() + 16, &s1, 4);
    std::memcpy(buffer.data() + 20, &s2, 4);

    float r = (col_f_dc_0[i] * SH_C0) + 0.5f;
    float g = (col_f_dc_1[i] * SH_C0) + 0.5f;
    float b = (col_f_dc_2[i] * SH_C0) + 0.5f;

    float a = 1.0f / (1.0f + std::exp(-col_opacity[i]));

    buffer[24] = float2UInt8(r);
    buffer[25] = float2UInt8(g);
    buffer[26] = float2UInt8(b);
    buffer[27] = float2UInt8(a);

    buffer[28] = float2UInt8((col_rot_0[i] + 1.0f) * 0.5f);
    buffer[29] = float2UInt8((col_rot_1[i] + 1.0f) * 0.5f);
    buffer[30] = float2UInt8((col_rot_2[i] + 1.0f) * 0.5f);
    buffer[31] = float2UInt8((col_rot_3[i] + 1.0f) * 0.5f);

    ofs.write(reinterpret_cast<const char*>(buffer.data()), 32);

    if ((i + 1) % 1000 == 0) {
      ofs.flush();
    }
  }
  ofs.close();
}

}  // namespace splat
