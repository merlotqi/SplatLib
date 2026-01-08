#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Camera {
 public:
  Camera();

  void update(float deltaTime);

  void rotate(float dx, float dy);
  void pan(float dx, float dy);
  void zoom(float delta);
  void orbit(float dx, float dy);

  void reset();
  void lookAt(const glm::vec3& target, float distance = 5.0f);
  void fitToBox(const glm::vec3& min, const glm::vec3& max);

  glm::mat4 getViewMatrix() const;
  glm::mat4 getProjectionMatrix(float aspect) const;
  glm::vec3 getPosition() const { return position_; }
  glm::vec3 getTarget() const { return target_; }

  void setPosition(const glm::vec3& pos);
  void setTarget(const glm::vec3& target);
  void setDistance(float distance);
  void setFOV(float fov) { fov_ = fov; }

  float fov_ = 60.0f;
  float near_plane_ = 0.1f;
  float far_plane_ = 10000.0f;
  float move_speed_ = 5.0f;
  float rotate_speed_ = 0.5f;
  float zoom_speed_ = 2.0f;

 private:
  void updateVectors();

 private:
  glm::vec3 position_;
  glm::vec3 target_;
  glm::vec3 up_;

  float distance_ = 5.0f;
  float azimuth_ = 0.0f;
  float elevation_ = 30.0f;
};
