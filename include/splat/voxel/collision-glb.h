/***********************************************************************************
 *
 * splat - A C++ library for reading and writing 3D Gaussian Splatting (splat) files.
 *
 * This file is part of splat.
 *
 * splat is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 ***********************************************************************************/

#pragma once

#include <absl/types/span.h>

#include <cstdint>
#include <vector>

namespace splat {

/**
 * @brief Build a minimal GLB (glTF 2.0 binary) with one triangle mesh.
 *
 * Contains only POSITION (VEC3 float) and triangle indices (uint32). Matches
 * splat-transform/lib/voxel/collision-glb.ts layout (JSON/BIN chunks, padding).
 *
 * @param positions Vertex positions, 3 floats per vertex
 * @param indices Triangle indices (3 per triangle)
 * @return GLB bytes
 * @throws std::invalid_argument if spans have wrong length
 */
std::vector<uint8_t> buildCollisionGlb(absl::Span<const float> positions, absl::Span<const uint32_t> indices);

}  // namespace splat
