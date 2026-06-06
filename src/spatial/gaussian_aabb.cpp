#include <absl/types/span.h>
#include <splat/models/splatcloud.h>
#include <splat/spatial/gaussian_aabb.h>
#include <splat/utils/logger.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <vector>

namespace splat {

GaussianExtentsResult computeGaussianExtents(const SplatCloud* dataTable) {
  const auto numRows = dataTable->getNumRows();

  // Get column data
  absl::Span<const float> x = dataTable->getColumnByName("x").asSpan<float>();
  absl::Span<const float> y = dataTable->getColumnByName("y").asSpan<float>();
  absl::Span<const float> z = dataTable->getColumnByName("z").asSpan<float>();
  absl::Span<const float> rx = dataTable->getColumnByName("rot_1").asSpan<float>();
  absl::Span<const float> ry = dataTable->getColumnByName("rot_2").asSpan<float>();
  absl::Span<const float> rz = dataTable->getColumnByName("rot_3").asSpan<float>();
  absl::Span<const float> rw = dataTable->getColumnByName("rot_0").asSpan<float>();
  absl::Span<const float> sx = dataTable->getColumnByName("scale_0").asSpan<float>();
  absl::Span<const float> sy = dataTable->getColumnByName("scale_1").asSpan<float>();
  absl::Span<const float> sz = dataTable->getColumnByName("scale_2").asSpan<float>();

  // Allocate output arrays
  std::vector<float> extentX(numRows);
  std::vector<float> extentY(numRows);
  std::vector<float> extentZ(numRows);

  // Scene bounds
  Eigen::Vector3f sceneMin(INFINITY, INFINITY, INFINITY);
  Eigen::Vector3f sceneMax(-INFINITY, -INFINITY, -INFINITY);

  // Reusable objects to avoid allocations in the loop
  Eigen::Vector3f position;
  Eigen::Quaternionf rotation;
  Eigen::Vector3f scale;
  Eigen::Matrix4f mat4;

  // Local AABB corners (3-sigma box centered at origin)
  std::array<Eigen::Vector3f, 8> localCorners;

  size_t invalidCount = 0;

  for (size_t i = 0; i < numRows; ++i) {
    // Get gaussian properties
    position << x[i], y[i], z[i];
    rotation.coeffs() << rx[i], ry[i], rz[i], rw[i];
    rotation.normalize();
    scale << std::exp(sx[i]), std::exp(sy[i]), std::exp(sz[i]);

    // Set local box corners to 3-sigma (Gaussians render out to 3-sigma)
    float halfX = scale.x() * 3.0f;
    float halfY = scale.y() * 3.0f;
    float halfZ = scale.z() * 3.0f;

    // Generate 8 corners of the local AABB
    localCorners[0] << -halfX, -halfY, -halfZ;
    localCorners[1] << halfX, -halfY, -halfZ;
    localCorners[2] << -halfX, halfY, -halfZ;
    localCorners[3] << halfX, halfY, -halfZ;
    localCorners[4] << -halfX, -halfY, halfZ;
    localCorners[5] << halfX, -halfY, halfZ;
    localCorners[6] << -halfX, halfY, halfZ;
    localCorners[7] << halfX, halfY, halfZ;

    // Create transformation matrix (rotation + translation, no scale since it's in localCorners already)
    mat4.setIdentity();
    mat4.block<3, 3>(0, 0) = rotation.toRotationMatrix();
    mat4.block<3, 1>(0, 3) = position;

    // Transform corners to world space and compute AABB
    Eigen::Vector3f worldMin(INFINITY, INFINITY, INFINITY);
    Eigen::Vector3f worldMax(-INFINITY, -INFINITY, -INFINITY);

    for (const auto& corner : localCorners) {
      Eigen::Vector3f worldCorner = (mat4 * Eigen::Vector4f(corner.x(), corner.y(), corner.z(), 1.0f)).head<3>();

      worldMin = worldMin.cwiseMin(worldCorner);
      worldMax = worldMax.cwiseMax(worldCorner);
    }

    // Get the half-extents of the world-space AABB
    Eigen::Vector3f halfExtents = (worldMax - worldMin) * 0.5f;

    // Validate
    if (!halfExtents.allFinite()) {
      // Store zero extents for invalid Gaussians
      extentX[i] = 0;
      extentY[i] = 0;
      extentZ[i] = 0;
      invalidCount++;
      continue;
    }

    // Store half-extents
    extentX[i] = halfExtents.x();
    extentY[i] = halfExtents.y();
    extentZ[i] = halfExtents.z();

    // Update scene bounds (AABB = position +/- halfExtents)
    Eigen::Vector3f minPos = position - halfExtents;
    Eigen::Vector3f maxPos = position + halfExtents;

    sceneMin = sceneMin.cwiseMin(minPos);
    sceneMax = sceneMax.cwiseMax(maxPos);
  }

  if (invalidCount > 0) {
    LOG_WARN("Skipped %zu Gaussians with invalid scale/rotation values", invalidCount);
  }

  auto cols = {Column{"extent_x", extentX}, Column{"extent_y", extentY}, Column{"extent_z", extentZ}};
  auto extentsTable = std::make_unique<SplatCloud>(cols);

  GaussianExtentsResult result;
  result.extents = std::make_unique<SplatCloud>(cols);
  result.invalidCount = invalidCount;
  result.sceneBounds.min = sceneMin;
  result.sceneBounds.max = sceneMax;

  return result;
}

}  // namespace splat
