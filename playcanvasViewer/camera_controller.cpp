#include "camera_controller.h"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <vector>

namespace playcanvas_viewer {

static Eigen::Matrix4f makeLookAt(const Eigen::Vector3f& eye, const Eigen::Vector3f& center, const Eigen::Vector3f& up) {
    Eigen::Vector3f f = (center - eye).normalized();
    Eigen::Vector3f s = f.cross(up).normalized();
    Eigen::Vector3f u = s.cross(f);
    Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
    mat.block<1, 3>(0, 0) = s.transpose();
    mat.block<1, 3>(1, 0) = u.transpose();
    mat.block<1, 3>(2, 0) = -f.transpose();
    mat(0, 3) = -s.dot(eye);
    mat(1, 3) = -u.dot(eye);
    mat(2, 3) = f.dot(eye);
    return mat;
}

static Eigen::Matrix4f makePerspective(float fovy, float aspect, float zNear, float zFar) {
    float tanHalfFovy = std::tan(fovy / 2.0f);
    Eigen::Matrix4f mat = Eigen::Matrix4f::Zero();
    mat(0, 0) = 1.0f / (aspect * tanHalfFovy);
    mat(1, 1) = 1.0f / tanHalfFovy;
    mat(2, 2) = -(zFar + zNear) / (zFar - zNear);
    mat(2, 3) = -2.0f * zFar * zNear / (zFar - zNear);
    mat(3, 2) = -1.0f;
    return mat;
}

static std::pair<float, float> robustRange(std::vector<float>& values) {
    std::sort(values.begin(), values.end());
    if (values.empty()) {
        return {0.0f, 0.0f};
    }

    size_t trim = 0;
    if (values.size() >= 1000) {
        trim = std::max<size_t>(1, values.size() / 1000);
    }

    const size_t low = std::min(trim, values.size() - 1);
    const size_t high = std::max(low, values.size() - 1 - trim);
    return {values[low], values[high]};
}

static float robustScalePadding(std::vector<float>& values) {
    std::sort(values.begin(), values.end());
    if (values.empty()) {
        return 0.0f;
    }

    size_t index = values.size() - 1;
    if (values.size() >= 1000) {
        index = std::min(values.size() - 1, values.size() - 1 - values.size() / 1000);
    }

    return values[index] * 2.0f;
}

SceneBounds computeSceneBounds(const splat::visualization::GSplatData& data) {
    SceneBounds bounds;
    if (data.centers.empty()) {
        bounds.center = Eigen::Vector3f(0, 0, 0);
        bounds.radius = 1.0f;
        return bounds;
    }
    std::vector<float> xs;
    std::vector<float> ys;
    std::vector<float> zs;
    std::vector<float> maxScales;
    xs.reserve(data.centers.size());
    ys.reserve(data.centers.size());
    zs.reserve(data.centers.size());
    maxScales.reserve(data.centers.size());

    for (size_t i = 0; i < data.centers.size(); ++i) {
        const auto& v = data.centers[i];
        const auto& s = data.scales[i];
        const float maxScale = std::max({s.x, s.y, s.z});
        if (std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && std::isfinite(maxScale)) {
            xs.push_back(v.x);
            ys.push_back(v.y);
            zs.push_back(v.z);
            maxScales.push_back(maxScale);
        }
    }

    if (xs.empty()) {
        bounds.center = Eigen::Vector3f(0, 0, 0);
        bounds.radius = 1.0f;
        return bounds;
    }

    const auto [minX, maxX] = robustRange(xs);
    const auto [minY, maxY] = robustRange(ys);
    const auto [minZ, maxZ] = robustRange(zs);
    const float padding = robustScalePadding(maxScales);
    const Eigen::Vector3f minPt(minX - padding, minY - padding, minZ - padding);
    const Eigen::Vector3f maxPt(maxX + padding, maxY + padding, maxZ + padding);

    bounds.center = (minPt + maxPt) * 0.5f;
    const Eigen::Vector3f halfExtents = (maxPt - minPt) * 0.5f;
    bounds.radius = std::max(0.01f, halfExtents.norm());
    return bounds;
}

void CameraController::resetToBounds(const SceneBounds& bounds) {
    m_target = bounds.center;
    const float aspect = static_cast<float>(std::max(m_width, 1)) / static_cast<float>(std::max(m_height, 1));
    const float verticalHalfFov = m_fovYRadians * 0.5f;
    const float horizontalHalfFov = std::atan(std::tan(verticalHalfFov) * aspect);
    const float fitHalfFov = std::max(0.001f, std::min(verticalHalfFov, horizontalHalfFov));
    float dist = (bounds.radius * 1.5f) / std::sin(fitHalfFov);
    dist = std::max(dist, 1.0f);
    m_position = bounds.center + Eigen::Vector3f(0, 0, dist);
    m_near = std::max(std::min(bounds.radius * 0.0005f, 0.01f), 0.0001f);
    m_far = std::max({dist + bounds.radius * 4.0f, bounds.radius * 32.0f, 10.0f});
    m_flySpeed = std::max(bounds.radius * 0.5f, 0.25f);
    m_minOrbitDistance = std::max({bounds.radius * 0.001f, m_near * 10.0f, 0.001f});
    m_minDollyStep = std::max(bounds.radius * 0.02f, 0.01f);
    m_changed = true;
}

void CameraController::orbit(float deltaX, float deltaY) {
    Eigen::Vector3f offset = m_position - m_target;
    float radius = offset.norm();
    if (radius < 1e-6f) return;
    float yaw = -deltaX * 0.005f;
    float pitch = -deltaY * 0.005f;
    Eigen::Vector3f up = m_up;
    Eigen::Vector3f fwd = (m_target - m_position).normalized();
    Eigen::Vector3f right = fwd.cross(up);
    if (right.norm() < 1e-6f) right = Eigen::Vector3f(1,0,0); // degeneracy guard
    right.normalize();
    Eigen::AngleAxisf yawRot(yaw, up);
    Eigen::AngleAxisf pitchRot(pitch, right);
    offset = yawRot * offset;
    offset = pitchRot * offset;
    m_position = m_target + offset;
    m_changed = true;
}

void CameraController::dolly(float amount) {
    Eigen::Vector3f offset = m_position - m_target;
    const float distance = offset.norm();
    if (distance < 1e-6f) {
        return;
    }

    const Eigen::Vector3f forward = (m_target - m_position) / distance;
    const float stepMagnitude = std::max(distance * 0.12f, m_minDollyStep) * std::abs(amount);

    if (amount > 0.0f) {
        const float desiredDistance = distance - stepMagnitude;
        if (desiredDistance < m_minOrbitDistance) {
            const float targetAdvance = m_minOrbitDistance - desiredDistance;
            m_target += forward * targetAdvance;
            m_position = m_target - forward * m_minOrbitDistance;
        } else {
            m_position += forward * stepMagnitude;
        }
    } else {
        m_position -= forward * stepMagnitude;
    }
    m_changed = true;
}

void CameraController::fly(const CameraInputState& input, float deltaSeconds) {
    if (!(input.forward || input.backward || input.left || input.right || input.up || input.down)) return;
    float dt = std::max(deltaSeconds, 0.0f);
    Eigen::Vector3f fwd = (m_target - m_position).normalized();
    Eigen::Vector3f right = fwd.cross(m_up).normalized();
    Eigen::Vector3f up = m_up;
    float fb = (input.forward ? 1.0f : 0.0f) - (input.backward ? 1.0f : 0.0f);
    float lr = (input.right ? 1.0f : 0.0f) - (input.left ? 1.0f : 0.0f);
    float ud = (input.up ? 1.0f : 0.0f) - (input.down ? 1.0f : 0.0f);
    Eigen::Vector3f move = fwd * fb + right * lr + up * ud;
    if (move.norm() > 1e-6f) {
        move.normalize();
        move *= m_flySpeed * dt;
        m_position += move;
        m_target += move;
        m_changed = true;
    }
}

void CameraController::resize(int width, int height) {
    m_width = std::max(width, 1);
    m_height = std::max(height, 1);
    m_changed = true;
}

Eigen::Matrix4f CameraController::viewMatrix() const {
    return makeLookAt(m_position, m_target, m_up);
}

Eigen::Matrix4f CameraController::projectionMatrix() const {
    float aspect = float(m_width) / float(m_height);
    return makePerspective(m_fovYRadians, aspect, m_near, m_far);
}

Eigen::Vector3f CameraController::position() const {
    return m_position;
}

Eigen::Vector3f CameraController::forward() const {
    return (m_target - m_position).normalized();
}

float CameraController::nearPlane() const {
    return m_near;
}

float CameraController::farPlane() const {
    return m_far;
}

bool CameraController::changed() const {
    return m_changed;
}

void CameraController::clearChanged() {
    m_changed = false;
}

} // namespace playcanvas_viewer
