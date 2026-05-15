#include "gsplat_renderer.h"
#include "gsplat_shader_sources.h"
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace playcanvas_viewer {

namespace {

GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    if (shader == 0) {
        throw std::runtime_error("glCreateShader failed");
    }
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        char log[1024];
        glGetShaderInfoLog(shader, 1024, nullptr, log);
        glDeleteShader(shader);
        throw std::runtime_error(std::string("Shader compile error: ") + log);
    }
    return shader;
}

GLuint buildProgram(const char* vs, const char* fs) {
    GLuint v = 0, f = 0, prog = 0;
    try {
        v = compileShader(GL_VERTEX_SHADER, vs);
        f = compileShader(GL_FRAGMENT_SHADER, fs);
        prog = glCreateProgram();
        if (prog == 0) {
            if (v) glDeleteShader(v);
            if (f) glDeleteShader(f);
            throw std::runtime_error("glCreateProgram failed");
        }
        glAttachShader(prog, v);
        glAttachShader(prog, f);
        glLinkProgram(prog);
        GLint status = 0;
        glGetProgramiv(prog, GL_LINK_STATUS, &status);
        if (!status) {
            char log[1024];
            glGetProgramInfoLog(prog, 1024, nullptr, log);
            glDeleteShader(v);
            glDeleteShader(f);
            glDeleteProgram(prog);
            throw std::runtime_error(std::string("Program link error: ") + log);
        }
        glDeleteShader(v);
        glDeleteShader(f);
        return prog;
    } catch (...) {
        if (v) glDeleteShader(v);
        if (f) glDeleteShader(f);
        if (prog) glDeleteProgram(prog);
        throw;
    }
}

inline void setMat4(GLint location, const Eigen::Matrix4f& mat) {
    if (location >= 0)
        glUniformMatrix4fv(location, 1, GL_FALSE, mat.data());
}

inline void setVec4(GLint location, float x, float y, float z, float w) {
    if (location >= 0)
        glUniform4f(location, x, y, z, w);
}

inline void setFloat(GLint location, float v) {
    if (location >= 0)
        glUniform1f(location, v);
}

} // namespace

GSplatRenderer::GSplatRenderer() {}

GSplatRenderer::~GSplatRenderer() {
    if (vao_) glDeleteVertexArrays(1, &vao_);
    if (cornerBuffer_) glDeleteBuffers(1, &cornerBuffer_);
    if (instanceBuffer_) glDeleteBuffers(1, &instanceBuffer_);
    if (program_) glDeleteProgram(program_);
}

void GSplatRenderer::ensureProgram() {
    if (program_) return;
    program_ = buildProgram(playcanvas_viewer::kGSplatVertexShader, playcanvas_viewer::kGSplatFragmentShader);
}

void GSplatRenderer::ensureBuffers() {
    if (vao_) return;
    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);

    // Corner buffer (quad)
    static const float corners[8] = {-1,-1, 1,-1, -1,1, 1,1};
    glGenBuffers(1, &cornerBuffer_);
    glBindBuffer(GL_ARRAY_BUFFER, cornerBuffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(corners), corners, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);

    // Instance buffer
    glGenBuffers(1, &instanceBuffer_);
    glBindBuffer(GL_ARRAY_BUFFER, instanceBuffer_);
    size_t stride = sizeof(InstanceData);
    // 1: center (vec3)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(InstanceData, center));
    glVertexAttribDivisor(1, 1);
    // 2: rotation (vec4)
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(InstanceData, rotation));
    glVertexAttribDivisor(2, 1);
    // 3: scale (vec3)
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(InstanceData, scale));
    glVertexAttribDivisor(3, 1);
    // 4: color (vec4)
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, stride, (void*)offsetof(InstanceData, color));
    glVertexAttribDivisor(4, 1);

    glBindVertexArray(0);
}

void GSplatRenderer::setData(const GSplatRenderData& data) {
    data_ = data;
    order_.resize(data.centers.size());
    std::iota(order_.begin(), order_.end(), 0u);
    sortedInstances_.resize(data.centers.size());
    instancesDirty_ = true;
    orderInitialized_ = false;
}

bool GSplatRenderer::needsInstanceOrderUpdate(const CameraController& camera, bool sortBackToFront) const {
    if (!orderInitialized_ || lastSortBackToFront_ != sortBackToFront) {
        return true;
    }
    if (!sortBackToFront) {
        return false;
    }

    const Eigen::Vector3f positionDelta = camera.position() - lastSortPosition_;
    const float forwardDot = std::clamp(camera.forward().dot(lastSortForward_), -1.0f, 1.0f);
    return positionDelta.squaredNorm() > 1e-4f || forwardDot < 0.9995f;
}

void GSplatRenderer::updateSortedInstances(const CameraController& camera, bool sortBackToFront) {
    if (data_.centers.empty()) return;
    if (!needsInstanceOrderUpdate(camera, sortBackToFront)) return;

    if (sortBackToFront) {
        std::sort(order_.begin(), order_.end(), [&](uint32_t a, uint32_t b) {
            Eigen::Vector3f ca = {data_.centers[a].x, data_.centers[a].y, data_.centers[a].z};
            Eigen::Vector3f cb = {data_.centers[b].x, data_.centers[b].y, data_.centers[b].z};
            float da = (ca - camera.position()).dot(camera.forward());
            float db = (cb - camera.position()).dot(camera.forward());
            return da > db;
        });
    } else {
        std::iota(order_.begin(), order_.end(), 0u);
    }
    for (size_t i = 0; i < order_.size(); ++i) {
        uint32_t idx = order_[i];
        const auto& c = data_.centers[idx];
        const auto& r = data_.rotations[idx];
        const auto& s = data_.scales[idx];
        const auto& col = data_.colors[idx];
        sortedInstances_[i].center = {c.x, c.y, c.z};
        sortedInstances_[i].rotation = {r.x, r.y, r.z, r.w};
        sortedInstances_[i].scale = {s.x, s.y, s.z};
        sortedInstances_[i].color = {col.x, col.y, col.z, col.w};
    }
    orderInitialized_ = true;
    lastSortBackToFront_ = sortBackToFront;
    lastSortPosition_ = camera.position();
    lastSortForward_ = camera.forward();
    instancesDirty_ = true;
}

void GSplatRenderer::uploadInstances() {
    if (!instancesDirty_) return;
    glBindBuffer(GL_ARRAY_BUFFER, instanceBuffer_);
    glBufferData(GL_ARRAY_BUFFER, sortedInstances_.size() * sizeof(InstanceData), sortedInstances_.data(), GL_DYNAMIC_DRAW);
    instancesDirty_ = false;
}

GLint GSplatRenderer::uniformLocation(const char* name) {
    return glGetUniformLocation(program_, name);
}

void GSplatRenderer::render(const CameraController& camera, int width, int height, const GSplatRenderOptions& options) {
    if (data_.centers.empty()) return;
    ensureProgram();
    ensureBuffers();
    width = std::max(width, 1);
    height = std::max(height, 1);
    updateSortedInstances(camera, options.sortBackToFront);
    uploadInstances();

    // Save GL state
    GLboolean blendEnabled = glIsEnabled(GL_BLEND);
    GLboolean cullEnabled = glIsEnabled(GL_CULL_FACE);
    GLboolean depthTestEnabled = glIsEnabled(GL_DEPTH_TEST);
    GLboolean depthMask = GL_TRUE;
    glGetBooleanv(GL_DEPTH_WRITEMASK, &depthMask);
    GLint blendSrcRGB = 0, blendDstRGB = 0, blendSrcA = 0, blendDstA = 0;
    glGetIntegerv(GL_BLEND_SRC_RGB, &blendSrcRGB);
    glGetIntegerv(GL_BLEND_DST_RGB, &blendDstRGB);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &blendSrcA);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &blendDstA);

    glUseProgram(program_);
    glBindVertexArray(vao_);

    setMat4(uniformLocation("uView"), camera.viewMatrix());
    setMat4(uniformLocation("uProjection"), camera.projectionMatrix());
    setVec4(uniformLocation("uViewport"), float(width), float(height), 1.0f/float(width), 1.0f/float(height));
    setVec4(uniformLocation("uCameraParams"), 1.0f / camera.farPlane(), camera.farPlane(), camera.nearPlane(), 0.0f); // perspective first version
    setFloat(uniformLocation("uGlobalOpacity"), options.globalOpacity);
    setFloat(uniformLocation("uSizeScale"), options.sizeScale);
    setFloat(uniformLocation("uMinPixelSize"), options.minPixelSize);
    setFloat(uniformLocation("uAlphaDiscardThreshold"), options.alphaDiscardThreshold);

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    glDepthMask(GL_FALSE);
    glEnable(GL_DEPTH_TEST);

    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(sortedInstances_.size()));

    // Restore GL state
    if (!blendEnabled) glDisable(GL_BLEND); else glEnable(GL_BLEND);
    if (cullEnabled) glEnable(GL_CULL_FACE); else glDisable(GL_CULL_FACE);
    if (depthTestEnabled) glEnable(GL_DEPTH_TEST); else glDisable(GL_DEPTH_TEST);
    glDepthMask(depthMask);
    glBlendFuncSeparate(blendSrcRGB, blendDstRGB, blendSrcA, blendDstA);

    glBindVertexArray(0);
    glUseProgram(0);
}

} // namespace playcanvas_viewer
