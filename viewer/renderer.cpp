/**
 * @file renderer.cpp
 * @brief OpenGL renderer implementation for Gaussian splatting.
 */

#include "renderer.h"

#include <splat/models/data-table.h>

#include <iostream>

#include "shader.h"

namespace viewer {

/** Vertex shader for point-based splat rendering */
static const char* splatVertSrc = R"(
#version 410 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aScale;
layout (location = 3) in vec4 aRotation;

uniform mat4 uVP;
uniform vec3 uCameraPos;

out vec3 vColor;
out vec3 vViewPos;
out vec3 vWorldPos;
out vec3 vScale;
out vec4 vRotation;

void main() {
    vColor = aColor;
    vScale = aScale;
    vRotation = aRotation;
    vWorldPos = aPos;
    vViewPos = (uVP * vec4(aPos, 1.0)).xyz;
    gl_Position = uVP * vec4(aPos, 1.0);
    gl_PointSize = max(2.0, 200.0 / length(vViewPos));
}
)";

/** Fragment shader for splat visualization */
static const char* splatFragSrc = R"(
#version 410 core

in vec3 vColor;
in vec3 vViewPos;
in vec3 vWorldPos;
in vec3 vScale;
in vec4 vRotation;

uniform float uOpacity;

out vec4 fragColor;

void main() {
    // Compute point coordinates relative to circle center
    vec2 screenPos = gl_PointCoord * 2.0 - 1.0;
    float dist = length(screenPos);

    // Soft Gaussian falloff
    float alpha = exp(-dist * dist * 2.0);

    if (alpha < 0.01) discard;

    fragColor = vec4(vColor * alpha, alpha * uOpacity);
}
)";

Renderer::Renderer() {}

Renderer::~Renderer() { unloadSplats(); }

void Renderer::initialize() {
  if (initialized) return;
  createShader();
  initialized = true;
}

void Renderer::createShader() { shader = std::make_unique<Shader>(splatVertSrc, splatFragSrc); }

void Renderer::loadSplats(const splat::DataTable& dataTable) {
  unloadSplats();

  size_t count = dataTable.getNumRows();
  if (count == 0) return;

  // Allocate buffers
  std::vector<GLfloat> positions(count * 3);
  std::vector<GLfloat> colors(count * 3);
  std::vector<GLfloat> scales(count * 3);
  std::vector<GLfloat> rotations(count * 4);

  // Extract columns
  const auto& xCol = dataTable.getColumnByName("x");
  const auto& yCol = dataTable.getColumnByName("y");
  const auto& zCol = dataTable.getColumnByName("z");
  const auto& f_dc0 = dataTable.getColumnByName("f_dc_0");
  const auto& f_dc1 = dataTable.getColumnByName("f_dc_1");
  const auto& f_dc2 = dataTable.getColumnByName("f_dc_2");
  const auto& opacity = dataTable.getColumnByName("opacity");
  const auto& scale0 = dataTable.getColumnByName("scale_0");
  const auto& scale1 = dataTable.getColumnByName("scale_1");
  const auto& scale2 = dataTable.getColumnByName("scale_2");
  const auto& rot0 = dataTable.getColumnByName("rot_0");
  const auto& rot1 = dataTable.getColumnByName("rot_1");
  const auto& rot2 = dataTable.getColumnByName("rot_2");
  const auto& rot3 = dataTable.getColumnByName("rot_3");

  const float SH_C0 = 0.28209479177387814f;

  auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };

  for (size_t i = 0; i < count; ++i) {
    // Position
    positions[i * 3 + 0] = xCol.getValue<float>(i);
    positions[i * 3 + 1] = yCol.getValue<float>(i);
    positions[i * 3 + 2] = zCol.getValue<float>(i);

    // Colors from SH DC coefficients
    colors[i * 3 + 0] = f_dc0.getValue<float>(i) * SH_C0 + 0.5f;
    colors[i * 3 + 1] = f_dc1.getValue<float>(i) * SH_C0 + 0.5f;
    colors[i * 3 + 2] = f_dc2.getValue<float>(i) * SH_C0 + 0.5f;

    // Clamp colors
    colors[i * 3 + 0] = std::clamp(colors[i * 3 + 0], 0.0f, 1.0f);
    colors[i * 3 + 1] = std::clamp(colors[i * 3 + 1], 0.0f, 1.0f);
    colors[i * 3 + 2] = std::clamp(colors[i * 3 + 2], 0.0f, 1.0f);

    // Scales (exp for log-scale)
    scales[i * 3 + 0] = std::exp(scale0.getValue<float>(i));
    scales[i * 3 + 1] = std::exp(scale1.getValue<float>(i));
    scales[i * 3 + 2] = std::exp(scale2.getValue<float>(i));

    // Rotation (normalize quaternion)
    float r0 = rot0.getValue<float>(i);
    float r1 = rot1.getValue<float>(i);
    float r2 = rot2.getValue<float>(i);
    float r3 = rot3.getValue<float>(i);
    float len = std::sqrt(r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3);
    if (len > 0.0f) {
      rotations[i * 4 + 0] = r0 / len;
      rotations[i * 4 + 1] = r1 / len;
      rotations[i * 4 + 2] = r2 / len;
      rotations[i * 4 + 3] = r3 / len;
    }
  }

  // Create VAO and buffers
  glGenVertexArrays(1, &splatData.vao);
  glBindVertexArray(splatData.vao);

  // Position buffer
  glGenBuffers(1, &splatData.posVbo);
  glBindBuffer(GL_ARRAY_BUFFER, splatData.posVbo);
  glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(GLfloat), positions.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(0);

  // Color buffer
  glGenBuffers(1, &splatData.colorVbo);
  glBindBuffer(GL_ARRAY_BUFFER, splatData.colorVbo);
  glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(GLfloat), colors.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(1);

  // Scale buffer
  glGenBuffers(1, &splatData.scaleVbo);
  glBindBuffer(GL_ARRAY_BUFFER, splatData.scaleVbo);
  glBufferData(GL_ARRAY_BUFFER, scales.size() * sizeof(GLfloat), scales.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(2);

  // Rotation buffer
  glGenBuffers(1, &splatData.rotationVbo);
  glBindBuffer(GL_ARRAY_BUFFER, splatData.rotationVbo);
  glBufferData(GL_ARRAY_BUFFER, rotations.size() * sizeof(GLfloat), rotations.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
  glEnableVertexAttribArray(3);

  glBindVertexArray(0);

  splatData.splatCount = count;
  splatData.isLoaded = true;

  std::cout << "Loaded " << count << " splats" << std::endl;
}

void Renderer::unloadSplats() {
  if (splatData.vao) glDeleteVertexArrays(1, &splatData.vao);
  if (splatData.posVbo) glDeleteBuffers(1, &splatData.posVbo);
  if (splatData.colorVbo) glDeleteBuffers(1, &splatData.colorVbo);
  if (splatData.scaleVbo) glDeleteBuffers(1, &splatData.scaleVbo);
  if (splatData.rotationVbo) glDeleteBuffers(1, &splatData.rotationVbo);

  splatData.vao = 0;
  splatData.posVbo = 0;
  splatData.colorVbo = 0;
  splatData.scaleVbo = 0;
  splatData.rotationVbo = 0;
  splatData.splatCount = 0;
  splatData.isLoaded = false;
}

void Renderer::render(const glm::mat4& vp, const glm::vec3& eyePosition) {
  if (!splatData.isLoaded || !shader) return;

  shader->use();
  shader->setMat4("uVP", vp);
  shader->setVec3("uCameraPos", eyePosition);
  shader->setFloat("uOpacity", 1.0f);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);

  glBindVertexArray(splatData.vao);
  glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(splatData.splatCount));
  glBindVertexArray(0);

  glDepthMask(GL_TRUE);
  glDisable(GL_BLEND);
}

}  // namespace viewer
