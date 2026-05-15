#pragma once
#include "camera_controller.h"
#include "gsplat_data_adapter.h"
#include <Eigen/Core>
#include <GL/glew.h>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <array>
#include <algorithm>
#include <numeric>

namespace playcanvas_viewer {

struct GSplatRenderOptions {
    float globalOpacity = 1.0f;
    float sizeScale = 1.0f;
    float minPixelSize = 2.0f;
    float alphaDiscardThreshold = 1.0f / 255.0f;
    bool sortBackToFront = true;
};

class GSplatRenderer {
public:
    GSplatRenderer();
    ~GSplatRenderer();
    GSplatRenderer(const GSplatRenderer&) = delete;
    GSplatRenderer& operator=(const GSplatRenderer&) = delete;

    void setData(const GSplatRenderData& data);
    void render(const CameraController& camera, int width, int height, const GSplatRenderOptions& options);

private:
    struct InstanceData {
        Vec3f center;
        Vec4f rotation;
        Vec3f scale;
        Vec4f color;
    };

    void ensureProgram();
    void ensureBuffers();
    void updateSortedInstances(const CameraController& camera, bool sortBackToFront);
    void uploadInstances();
    GLint uniformLocation(const char* name);
    // Helper setters for uniforms (guarded for loc < 0)
    friend inline void setMat4(GLint location, const Eigen::Matrix4f& mat);
    friend inline void setVec4(GLint location, float x, float y, float z, float w);
    friend inline void setFloat(GLint location, float v);

    GSplatRenderData data_;
    std::vector<uint32_t> order_;
    std::vector<InstanceData> sortedInstances_;
    GLuint vao_ = 0;
    GLuint cornerBuffer_ = 0;
    GLuint instanceBuffer_ = 0;
    GLuint program_ = 0;
    bool instancesDirty_ = true;
};

} // namespace playcanvas_viewer
