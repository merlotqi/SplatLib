/**
 * @file splat/voxel/collision-glb.h
 * @brief Minimal triangle mesh GLB for collision preview.
 *
 * References:
 * - [glTF 2.0](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
 */
 
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
