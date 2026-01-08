#include "camera.h"

#include <algorithm>
#include <cmath>

Camera::Camera() { reset(); }

void Camera::reset() {
  position_ = glm::vec3(0.0f, 0.0f, 5.0f);
  target_ = glm::vec3(0.0f, 0.0f, 0.0f);
  up_ = glm::vec3(0.0f, 1.0f, 0.0f);
  distance_ = 5.0f;
  azimuth_ = 0.0f;
  elevation_ = 30.0f;
  updateVectors();
}

void Camera::update(float deltaTime) {
  // 可以在这里添加平滑过渡等效果
}

void Camera::rotate(float dx, float dy) {
  azimuth_ += dx * rotate_speed_;
  elevation_ += dy * rotate_speed_;

  // 限制垂直角度
  elevation_ = std::clamp(elevation_, -89.0f, 89.0f);

  updateVectors();
}

void Camera::pan(float dx, float dy) {
  glm::vec3 right = glm::normalize(glm::cross(glm::normalize(target_ - position_), up_));
  glm::vec3 actual_up = glm::normalize(glm::cross(right, glm::normalize(target_ - position_)));

  float scale = distance_ * 0.001f * move_speed_;
  glm::vec3 delta = -right * dx * scale + actual_up * dy * scale;

  position_ += delta;
  target_ += delta;
}

void Camera::zoom(float delta) {
  distance_ *= (1.0f - delta * 0.1f * zoom_speed_);
  distance_ = std::max(distance_, 0.1f);
  updateVectors();
}

void Camera::orbit(float dx, float dy) { rotate(dx, dy); }

void Camera::lookAt(const glm::vec3& target, float distance) {
  target_ = target;
  distance_ = distance;
  updateVectors();
}

void Camera::fitToBox(const glm::vec3& min, const glm::vec3& max) {
  glm::vec3 center = (min + max) * 0.5f;
  glm::vec3 size = max - min;

  float max_size = std::max({size.x, size.y, size.z});
  if (max_size < 0.0001f) max_size = 1.0f;

  // 计算合适的距离
  float fov_rad = glm::radians(fov_);
  float distance = (max_size * 0.5f) / std::tan(fov_rad * 0.5f) * 1.5f;

  lookAt(center, std::max(distance, 1.0f));

  // 调整近远平面
  near_plane_ = distance * 0.01f;
  far_plane_ = distance * 100.0f;
}

glm::mat4 Camera::getViewMatrix() const { return glm::lookAt(position_, target_, up_); }

glm::mat4 Camera::getProjectionMatrix(float aspect) const {
  return glm::perspective(glm::radians(fov_), aspect, near_plane_, far_plane_);
}

void Camera::setPosition(const glm::vec3& pos) {
  position_ = pos;
  glm::vec3 dir = target_ - position_;
  distance_ = glm::length(dir);

  if (distance_ > 0.0f) {
    dir = glm::normalize(dir);
    elevation_ = glm::degrees(std::asin(dir.y));
    azimuth_ = glm::degrees(std::atan2(-dir.z, -dir.x));
  }
}

void Camera::setTarget(const glm::vec3& target) {
  target_ = target;
  updateVectors();
}

void Camera::setDistance(float distance) {
  distance_ = std::max(distance, 0.1f);
  updateVectors();
}

void Camera::updateVectors() {
  // 将球坐标转换为笛卡尔坐标
  float az_rad = glm::radians(azimuth_);
  float el_rad = glm::radians(elevation_);

  position_.x = target_.x + distance_ * std::cos(el_rad) * std::cos(az_rad);
  position_.y = target_.y + distance_ * std::sin(el_rad);
  position_.z = target_.z + distance_ * std::cos(el_rad) * std::sin(az_rad);

  // 更新up向量（保持垂直）
  glm::vec3 forward = glm::normalize(target_ - position_);
  glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
  up_ = glm::normalize(glm::cross(right, forward));
}
