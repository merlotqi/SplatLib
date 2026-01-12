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

#include <splat/io/compressed_chunk.h>
#include <splat/maths/maths.h>

#include <Eigen/Dense>
#include <algorithm>

namespace splat {

struct MinMax {
  float min = std::numeric_limits<float>::infinity();
  float max = -std::numeric_limits<float>::infinity();
};

static MinMax calcMinMax(const std::vector<float>& data) {
  if (data.empty()) return {0.0f, 0.0f};

  float minVal = data[0];
  float maxVal = data[0];
  for (size_t i = 1; i < data.size(); ++i) {
    minVal = std::min(minVal, data[i]);
    maxVal = std::max(maxVal, data[i]);
  }
  return {minVal, maxVal};
}

static float normalize(float x, float min, float max) {
  if (x <= min) return 0.0f;
  if (x >= max) return 1.0f;

  float range = max - min;
  return (range < 0.00001f) ? 0.0f : (x - min) / range;
}

static uint32_t packUnorm(float value, int bits) {
  uint32_t t = (1u << bits) - 1;
  float scaled = value * t + 0.5f;
  return std::max(0u, std::min(t, static_cast<uint32_t>(std::floor(scaled))));
}

static uint32_t pack111011(float x, float y, float z) {
  return (packUnorm(x, 11) << 21) | (packUnorm(y, 10) << 11) | (packUnorm(z, 11));
}

static uint32_t pack8888(float x, float y, float z, float w) {
  return (packUnorm(x, 8) << 24) |  // R
         (packUnorm(y, 8) << 16) |  // G
         (packUnorm(z, 8) << 8) |   // B
         (packUnorm(w, 8));         // A
}

static uint32_t packRot(float x, float y, float z, float w) {
  Eigen::Quaternionf q(w, x, y, z);
  q.normalize();

  Eigen::Vector3f v = q.vec();
  float a[4] = {v.x(), v.y(), v.z(), q.w()};

  int largest = 0;
  for (int i = 1; i < 4; ++i) {
    if (std::abs(a[i]) > std::abs(a[largest])) {
      largest = i;
    }
  }

  if (a[largest] < 0) {
    for (int i = 0; i < 4; ++i) a[i] = -a[i];
  }

  uint32_t result = largest;

  const float norm = std::sqrt(2.0f) * 0.5f;
  for (int i = 0; i < 4; ++i) {
    if (i != largest) {
      result = (result << 10) | packUnorm(a[i] * norm + 0.5f, 10);
    }
  }

  return result;
}

static const std::vector<std::string> members = {"x",       "y",      "z",      "scale_0", "scale_1",
                                                 "scale_2", "f_dc_0", "f_dc_1", "f_dc_2",  "opacity",
                                                 "rot_0",   "rot_1",  "rot_2",  "rot_3"};

CompressedChunk::CompressedChunk(size_t sz) : size(sz) {
  for (const auto& m : members) {
    this->data[m].resize(size);
  }
  this->chunkData.resize(18);
  this->position.resize(size);
  this->rotation.resize(size);
  this->scale.resize(size);
  this->color.resize(size);
}

void CompressedChunk::set(size_t index, const std::map<std::string, float>& dataMap) {
  if (index >= size) return;
  for (const auto& m : members) {
    if (dataMap.count(m)) {
      this->data[m][index] = dataMap.at(m);
    }
  }
}

void CompressedChunk::pack() {
  auto& x = data["x"];
  auto& y = data["y"];
  auto& z = data["z"];
  auto& scale_0 = data["scale_0"];
  auto& scale_1 = data["scale_1"];
  auto& scale_2 = data["scale_2"];
  auto& rot_0 = data["rot_0"];
  auto& rot_1 = data["rot_1"];
  auto& rot_2 = data["rot_2"];
  auto& rot_3 = data["rot_3"];
  auto& f_dc_0 = data["f_dc_0"];
  auto& f_dc_1 = data["f_dc_1"];
  auto& f_dc_2 = data["f_dc_2"];
  auto& opacity = data["opacity"];

  MinMax px = calcMinMax(x);
  MinMax py = calcMinMax(y);
  MinMax pz = calcMinMax(z);

  MinMax sx = calcMinMax(scale_0);
  MinMax sy = calcMinMax(scale_1);
  MinMax sz = calcMinMax(scale_2);

  auto clamp = [](float v) { return std::max(-20.0f, std::min(20.0f, v)); };
  sx.min = clamp(sx.min);
  sx.max = clamp(sx.max);
  sy.min = clamp(sy.min);
  sy.max = clamp(sy.max);
  sz.min = clamp(sz.min);
  sz.max = clamp(sz.max);

  const float SH_C0 = 0.28209479177387814f;
  for (size_t i = 0; i < size; ++i) {
    f_dc_0[i] = f_dc_0[i] * SH_C0 + 0.5f;
    f_dc_1[i] = f_dc_1[i] * SH_C0 + 0.5f;
    f_dc_2[i] = f_dc_2[i] * SH_C0 + 0.5f;
  }

  MinMax cr = calcMinMax(f_dc_0);
  MinMax cg = calcMinMax(f_dc_1);
  MinMax cb = calcMinMax(f_dc_2);

  for (size_t i = 0; i < size; ++i) {
    // Position (11, 10, 11 bits)
    position[i] =
        pack111011(normalize(x[i], px.min, px.max), normalize(y[i], py.min, py.max), normalize(z[i], pz.min, pz.max));

    // Rotation (2, 10, 10, 10 bits)
    rotation[i] = packRot(rot_0[i], rot_1[i], rot_2[i], rot_3[i]);

    // Scale (11, 10, 11 bits)
    scale[i] = pack111011(normalize(scale_0[i], sx.min, sx.max), normalize(scale_1[i], sy.min, sy.max),
                          normalize(scale_2[i], sz.min, sz.max));

    // Color (8, 8, 8, 8 bits) - Opacity uses sigmoid before packing
    color[i] = pack8888(normalize(f_dc_0[i], cr.min, cr.max), normalize(f_dc_1[i], cg.min, cg.max),
                        normalize(f_dc_2[i], cb.min, cb.max), sigmoid(opacity[i]));
  }

  chunkData[0] = px.min;
  chunkData[1] = py.min;
  chunkData[2] = pz.min;
  chunkData[3] = px.max;
  chunkData[4] = py.max;
  chunkData[5] = pz.max;

  chunkData[6] = sx.min;
  chunkData[7] = sy.min;
  chunkData[8] = sz.min;
  chunkData[9] = sx.max;
  chunkData[10] = sy.max;
  chunkData[11] = sz.max;

  chunkData[12] = cr.min;
  chunkData[13] = cg.min;
  chunkData[14] = cb.min;
  chunkData[15] = cr.max;
  chunkData[16] = cg.max;
  chunkData[17] = cb.max;
}

}  // namespace splat
