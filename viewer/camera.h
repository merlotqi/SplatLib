/**
 * @file camera.h
 * @brief 3D camera with orbit controls for splat viewing.
 */

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace viewer {

/**
 * @brief Orbit camera for 3D scene navigation.
 *
 * Provides rotation around a target point, zoom via scroll,
 * and pan with right mouse button. Uses spherical coordinates
 * for intuitive camera control.
 */
class Camera {
 public:
  glm::vec3 target{0.0f};          ///< Camera target (look-at point)
  float radius{5.0f};              ///< Distance from target
  float theta{0.0f};               ///< Horizontal rotation (radians)
  float phi{1.0f};                 ///< Vertical rotation (radians), range (0, PI)
  glm::vec3 up{0.0f, 1.0f, 0.0f};  ///< Up direction

  /**
   * @brief Construct camera at default position.
   */
  Camera() = default;

  /**
   * @brief Set camera to look at a specific target.
   * @param newTarget Target point
   */
  void setTarget(const glm::vec3& newTarget) { target = newTarget; }

  /**
   * @brief Set camera distance from target.
   * @param newRadius Distance (clamped to [1.0, 1000.0])
   */
  void setRadius(float newRadius) { radius = glm::clamp(newRadius, 1.0f, 1000.0f); }

  /**
   * @brief Rotate camera by delta angles.
   * @param dTheta Horizontal rotation delta (radians)
   * @param dPhi Vertical rotation delta (radians)
   */
  void rotate(float dTheta, float dPhi) {
    theta += dTheta;
    phi = glm::clamp(phi + dPhi, 0.01f, 3.14f - 0.01f);
  }

  /**
   * @brief Pan the camera target.
   * @param dx Horizontal pan amount
   * @param dy Vertical pan amount
   */
  void pan(float dx, float dy) {
    glm::vec3 right = glm::normalize(glm::cross(getFront(), up));
    target += right * dx * radius * 0.001f;
    target += up * dy * radius * 0.001f;
  }

  /**
   * @brief Zoom the camera (change radius).
   * @param delta Positive = zoom out, negative = zoom in
   */
  void zoom(float delta) {
    radius *= 1.0f + delta * 0.01f;
    radius = glm::clamp(radius, 1.0f, 1000.0f);
  }

  /**
   * @brief Get camera position from spherical coordinates.
   * @return Camera position in world space
   */
  glm::vec3 getPosition() const {
    float x = radius * std::sin(phi) * std::cos(theta);
    float y = radius * std::cos(phi);
    float z = radius * std::sin(phi) * std::sin(theta);
    return target + glm::vec3(x, y, z);
  }

  /**
   * @brief Get view direction vector.
   * @return Normalized front vector
   */
  glm::vec3 getFront() const { return glm::normalize(target - getPosition()); }

  /**
   * @brief Get right vector.
   * @return Normalized right vector
   */
  glm::vec3 getRight() const { return glm::normalize(glm::cross(getFront(), up)); }

  /**
   * @brief Get view matrix (world to camera).
   * @return 4x4 view matrix
   */
  glm::mat4 getViewMatrix() const { return glm::lookAt(getPosition(), target, up); }
};

}  // namespace viewer
