#include "camera_controller.h"

#include "gsplat_data.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using playcanvas_viewer::CameraController;
using playcanvas_viewer::SceneBounds;
using playcanvas_viewer::computeSceneBounds;
using splat::visualization::GSplatData;
using splat::visualization::Vec3f;

namespace {

bool nearlyEqual(float lhs, float rhs, float epsilon = 1e-4f) {
    return std::abs(lhs - rhs) <= epsilon;
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_computeSceneBounds_uses_half_diagonal_radius() {
    GSplatData data;
    data.centers = {
        Vec3f{-2.0f, -4.0f, -8.0f},
        Vec3f{ 2.0f,  4.0f,  8.0f}
    };
    data.scales = {
        Vec3f{0.0f, 0.0f, 0.0f},
        Vec3f{0.0f, 0.0f, 0.0f}
    };

    const SceneBounds bounds = computeSceneBounds(data);

    require(nearlyEqual(bounds.center.x(), 0.0f), "bounds center x should be zero");
    require(nearlyEqual(bounds.center.y(), 0.0f), "bounds center y should be zero");
    require(nearlyEqual(bounds.center.z(), 0.0f), "bounds center z should be zero");
    require(nearlyEqual(bounds.radius, std::sqrt(2.0f * 2.0f + 4.0f * 4.0f + 8.0f * 8.0f)), "bounds radius should be half diagonal");
}

void test_computeSceneBounds_ignores_sparse_far_outliers_for_camera_framing() {
    GSplatData data;
    for (int i = 0; i < 2000; ++i) {
        const float x = static_cast<float>((i % 20) - 10);
        const float y = static_cast<float>(((i / 20) % 20) - 10);
        const float z = static_cast<float>(((i / 400) % 5) - 2);
        data.centers.push_back(Vec3f{x, y, z});
        data.scales.push_back(Vec3f{0.01f, 0.01f, 0.01f});
    }
    data.centers.push_back(Vec3f{10000.0f, 10000.0f, 10000.0f});
    data.scales.push_back(Vec3f{0.01f, 0.01f, 0.01f});

    const SceneBounds bounds = computeSceneBounds(data);

    require(bounds.radius < 40.0f, "camera framing radius should ignore sparse far outliers");
    require(bounds.center.norm() < 5.0f, "camera framing center should stay near the dense scene body");
}

void test_resetToBounds_matches_playcanvas_focus_distance() {
    CameraController camera;
    camera.resize(1280, 720);

    SceneBounds bounds;
    bounds.center = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
    bounds.radius = 10.0f;
    camera.resetToBounds(bounds);

    const float expectedDistance = 30.0f;
    const float actualDistance = (camera.position() - bounds.center).norm();

    require(nearlyEqual(actualDistance, expectedDistance), "reset distance should match PlayCanvas focus distance");
}

void test_resetToBounds_far_plane_covers_narrow_view_fit() {
    CameraController camera;
    camera.resize(16, 1280);

    SceneBounds bounds;
    bounds.center = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    bounds.radius = 10.0f;
    camera.resetToBounds(bounds);

    const float actualDistance = (camera.position() - bounds.center).norm();

    require(camera.farPlane() > actualDistance + bounds.radius, "far plane should cover fitted narrow view distance");
}

void test_resetToBounds_keeps_near_plane_suitable_for_close_inspection() {
    CameraController camera;

    SceneBounds bounds;
    bounds.center = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    bounds.radius = 5000.0f;
    camera.resetToBounds(bounds);

    require(camera.nearPlane() <= 0.05f, "near plane should allow close inspection");
}

void test_dolly_can_continue_past_initial_target() {
    CameraController camera;

    SceneBounds bounds;
    bounds.center = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    bounds.radius = 10.0f;
    camera.resetToBounds(bounds);

    for (int i = 0; i < 120; ++i) {
        camera.dolly(1.0f);
    }

    require(camera.position().z() < 0.0f, "dolly should continue past the initial target");
    require(std::isfinite(camera.forward().x()), "forward x should stay finite");
    require(std::isfinite(camera.forward().y()), "forward y should stay finite");
    require(std::isfinite(camera.forward().z()), "forward z should stay finite");
}

void test_dolly_keeps_useful_close_range_step_size() {
    CameraController camera;

    SceneBounds bounds;
    bounds.center = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    bounds.radius = 100.0f;
    camera.resetToBounds(bounds);

    for (int i = 0; i < 80; ++i) {
        camera.dolly(1.0f);
    }
    const float before = camera.position().z();
    camera.dolly(1.0f);
    const float after = camera.position().z();

    require(std::abs(after - before) > 0.5f, "close-range dolly should retain a useful world-space step");
}

void test_viewMatrix_uses_upright_screen_orientation() {
    CameraController camera;

    SceneBounds bounds;
    bounds.center = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    bounds.radius = 10.0f;
    camera.resetToBounds(bounds);

    const Eigen::Matrix4f view = camera.viewMatrix();
    const Eigen::Vector4f worldUp = view * Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f);
    const Eigen::Vector4f origin = view * Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f);

    require(worldUp.y() < origin.y(), "world +Y should map downward to match the PlayCanvas-style upright view for this data");
}

} // namespace

int main() {
    test_computeSceneBounds_uses_half_diagonal_radius();
    test_computeSceneBounds_ignores_sparse_far_outliers_for_camera_framing();
    test_resetToBounds_matches_playcanvas_focus_distance();
    test_resetToBounds_far_plane_covers_narrow_view_fit();
    test_resetToBounds_keeps_near_plane_suitable_for_close_inspection();
    test_dolly_can_continue_past_initial_target();
    test_dolly_keeps_useful_close_range_step_size();
    test_viewMatrix_uses_upright_screen_orientation();
    std::cout << "All PlaycanvasViewerCamera tests passed.\n";
    return 0;
}
