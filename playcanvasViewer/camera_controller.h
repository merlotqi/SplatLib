#pragma once
#include <Eigen/Core>
#include <splat/visualization/gsplat_data.h>

namespace playcanvas_viewer {

struct CameraInputState {
    bool forward = false;
    bool backward = false;
    bool left = false;
    bool right = false;
    bool down = false;
    bool up = false;
};

struct SceneBounds {
    Eigen::Vector3f center{0, 0, 0};
    float radius = 1.0f;
};

class CameraController {
public:
    void resetToBounds(const SceneBounds& bounds);
    void orbit(float deltaX, float deltaY);
    void dolly(float amount);
    void fly(const CameraInputState& input, float deltaSeconds);
    void resize(int width, int height);
    Eigen::Matrix4f viewMatrix() const;
    Eigen::Matrix4f projectionMatrix() const;
    Eigen::Vector3f position() const;
    Eigen::Vector3f forward() const;
    float nearPlane() const;
    float farPlane() const;
    bool changed() const;
    void clearChanged();

private:
    Eigen::Vector3f m_position{0, 0, 3};
    Eigen::Vector3f m_target{0, 0, 0};
    Eigen::Vector3f m_up{0, -1, 0};
    int m_width = 1280;
    int m_height = 720;
    static constexpr float kFovYRadians = 1.0471975512f; // 60 deg in radians
    float m_fovYRadians = kFovYRadians;
    float m_near = 0.01f;
    float m_far = 1000.0f;
    float m_flySpeed = 3.0f;
    float m_minOrbitDistance = 0.01f;
    float m_minDollyStep = 0.01f;
    bool m_changed = true;
};

SceneBounds computeSceneBounds(const splat::visualization::GSplatData& data);

} // namespace playcanvas_viewer
