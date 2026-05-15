#include <splat/visualization/gsplat_gl_renderer.h>
#include "gsplat_gl_shader_sources.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>

namespace splat::visualization {
namespace {

GLuint compileShader(GLenum type, const char* source) {
  const GLuint shader = glCreateShader(type);
  if (shader == 0) {
    throw std::runtime_error("glCreateShader failed");
  }

  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint status = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
  if (status == GL_TRUE) {
    return shader;
  }

  char log[2048] = {};
  glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
  glDeleteShader(shader);
  throw std::runtime_error(std::string("GSplat shader compile error: ") + log);
}

GLuint buildProgram() {
  GLuint vertexShader = 0;
  GLuint fragmentShader = 0;
  GLuint program = 0;
  try {
    vertexShader = compileShader(GL_VERTEX_SHADER, kGSplatGLVertexShader);
    fragmentShader = compileShader(GL_FRAGMENT_SHADER, kGSplatGLFragmentShader);
    program = glCreateProgram();
    if (program == 0) {
      throw std::runtime_error("glCreateProgram failed");
    }
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    GLint status = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
      char log[2048] = {};
      glGetProgramInfoLog(program, sizeof(log), nullptr, log);
      throw std::runtime_error(std::string("GSplat shader link error: ") + log);
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return program;
  } catch (...) {
    if (vertexShader != 0) glDeleteShader(vertexShader);
    if (fragmentShader != 0) glDeleteShader(fragmentShader);
    if (program != 0) glDeleteProgram(program);
    throw;
  }
}

void setMat4(GLint location, const Eigen::Matrix4f& matrix) {
  if (location >= 0) {
    glUniformMatrix4fv(location, 1, GL_FALSE, matrix.data());
  }
}

void setVec4(GLint location, float x, float y, float z, float w) {
  if (location >= 0) {
    glUniform4f(location, x, y, z, w);
  }
}

void setFloat(GLint location, float value) {
  if (location >= 0) {
    glUniform1f(location, value);
  }
}

}  // namespace

GSplatGLRenderer::~GSplatGLRenderer() { this->releaseGraphicsResources(); }

void GSplatGLRenderer::releaseGraphicsResources() {
  if (vao_ != 0) glDeleteVertexArrays(1, &vao_);
  if (cornerBuffer_ != 0) glDeleteBuffers(1, &cornerBuffer_);
  if (instanceBuffer_ != 0) glDeleteBuffers(1, &instanceBuffer_);
  if (program_ != 0) glDeleteProgram(program_);
  vao_ = 0;
  cornerBuffer_ = 0;
  instanceBuffer_ = 0;
  program_ = 0;
  instancesDirty_ = true;
  orderInitialized_ = false;
}

void GSplatGLRenderer::setData(const GSplatData& data) {
  data_ = data;
  order_.resize(data_.size());
  std::iota(order_.begin(), order_.end(), 0u);
  sortedInstances_.resize(data_.size());
  instancesDirty_ = true;
  orderInitialized_ = false;
}

void GSplatGLRenderer::ensureProgram() {
  if (program_ == 0) {
    program_ = buildProgram();
  }
}

void GSplatGLRenderer::ensureBuffers() {
  if (vao_ != 0) {
    return;
  }

  glGenVertexArrays(1, &vao_);
  glBindVertexArray(vao_);

  static constexpr float corners[8] = {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
  glGenBuffers(1, &cornerBuffer_);
  glBindBuffer(GL_ARRAY_BUFFER, cornerBuffer_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(corners), corners, GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);

  glGenBuffers(1, &instanceBuffer_);
  glBindBuffer(GL_ARRAY_BUFFER, instanceBuffer_);
  const auto stride = static_cast<GLsizei>(sizeof(InstanceData));

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(InstanceData, center)));
  glVertexAttribDivisor(1, 1);

  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(InstanceData, rotation)));
  glVertexAttribDivisor(2, 1);

  glEnableVertexAttribArray(3);
  glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(InstanceData, scale)));
  glVertexAttribDivisor(3, 1);

  glEnableVertexAttribArray(4);
  glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(InstanceData, color)));
  glVertexAttribDivisor(4, 1);

  glBindVertexArray(0);
}

bool GSplatGLRenderer::needsInstanceOrderUpdate(const GSplatGLFrameState& frame, bool sortBackToFront) const {
  if (!orderInitialized_ || lastSortBackToFront_ != sortBackToFront) {
    return true;
  }
  if (!sortBackToFront) {
    return false;
  }

  const Eigen::Vector3f positionDelta = frame.cameraPosition - lastSortPosition_;
  const float forwardDot = std::clamp(frame.cameraForward.normalized().dot(lastSortForward_.normalized()), -1.0f, 1.0f);
  return positionDelta.squaredNorm() > 1e-4f || forwardDot < 0.9995f;
}

void GSplatGLRenderer::updateSortedInstances(const GSplatGLFrameState& frame, bool sortBackToFront) {
  if (data_.empty() || !needsInstanceOrderUpdate(frame, sortBackToFront)) {
    return;
  }

  const Eigen::Vector3f cameraForward = frame.cameraForward.normalized();
  if (sortBackToFront) {
    std::sort(order_.begin(), order_.end(), [&](std::uint32_t lhs, std::uint32_t rhs) {
      const auto& left = data_.centers[lhs];
      const auto& right = data_.centers[rhs];
      const Eigen::Vector4f leftCenter(left.x, left.y, left.z, 1.0f);
      const Eigen::Vector4f rightCenter(right.x, right.y, right.z, 1.0f);
      const float leftDepth = (frame.view * leftCenter).z();
      const float rightDepth = (frame.view * rightCenter).z();
      return leftDepth < rightDepth;
    });
  } else {
    std::iota(order_.begin(), order_.end(), 0u);
  }

  for (std::size_t i = 0; i < order_.size(); ++i) {
    const std::uint32_t source = order_[i];
    sortedInstances_[i] = {data_.centers[source], data_.rotations[source], data_.scales[source], data_.colors[source]};
  }

  orderInitialized_ = true;
  lastSortBackToFront_ = sortBackToFront;
  lastSortPosition_ = frame.cameraPosition;
  lastSortForward_ = cameraForward;
  instancesDirty_ = true;
}

void GSplatGLRenderer::uploadInstances() {
  if (!instancesDirty_) {
    return;
  }
  glBindBuffer(GL_ARRAY_BUFFER, instanceBuffer_);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sortedInstances_.size() * sizeof(InstanceData)),
               sortedInstances_.data(), GL_DYNAMIC_DRAW);
  instancesDirty_ = false;
}

void GSplatGLRenderer::render(const GSplatGLFrameState& frame, const GSplatGLRenderOptions& options) {
  if (data_.empty()) {
    return;
  }

  this->ensureProgram();
  this->ensureBuffers();
  this->updateSortedInstances(frame, options.sortBackToFront);
  this->uploadInstances();

  glUseProgram(program_);
  glBindVertexArray(vao_);

  const int width = std::max(frame.width, 1);
  const int height = std::max(frame.height, 1);
  setMat4(glGetUniformLocation(program_, "uView"), frame.view);
  setMat4(glGetUniformLocation(program_, "uProjection"), frame.projection);
  setVec4(glGetUniformLocation(program_, "uViewport"), static_cast<float>(width), static_cast<float>(height),
          1.0f / static_cast<float>(width), 1.0f / static_cast<float>(height));
  setVec4(glGetUniformLocation(program_, "uCameraParams"), 1.0f / frame.farPlane, frame.farPlane, frame.nearPlane, 0.0f);
  setFloat(glGetUniformLocation(program_, "uGlobalOpacity"), options.globalOpacity);
  setFloat(glGetUniformLocation(program_, "uSizeScale"), options.sizeScale);
  setFloat(glGetUniformLocation(program_, "uMinPixelSize"), options.minPixelSize);
  setFloat(glGetUniformLocation(program_, "uMaxPixelSize"), options.maxPixelSize);
  setFloat(glGetUniformLocation(program_, "uAlphaDiscardThreshold"), options.alphaDiscardThreshold);
  setFloat(glGetUniformLocation(program_, "uClampColors"), options.clampColors ? 1.0f : 0.0f);

  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  if (options.depthTest) {
    glEnable(GL_DEPTH_TEST);
  } else {
    glDisable(GL_DEPTH_TEST);
  }
  glDepthMask(options.depthWrite ? GL_TRUE : GL_FALSE);
  glDisable(GL_CULL_FACE);

  glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, static_cast<GLsizei>(sortedInstances_.size()));

  glBindVertexArray(0);
  glUseProgram(0);
}

}  // namespace splat::visualization
