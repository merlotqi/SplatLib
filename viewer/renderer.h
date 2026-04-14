/**
 * @file renderer.h
 * @brief OpenGL renderer for Gaussian splat visualization.
 */

#pragma once

#include <GL/glew.h>

#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

namespace splat {
class DataTable;
}

namespace viewer {

class Shader;

/**
 * @brief Splat data buffers for GPU rendering.
 */
struct SplatData {
  GLuint vao = 0;          ///< Vertex array object
  GLuint posVbo = 0;       ///< Position buffer
  GLuint colorVbo = 0;     ///< Color buffer
  GLuint scaleVbo = 0;     ///< Scale buffer
  GLuint rotationVbo = 0;  ///< Rotation buffer
  size_t splatCount = 0;   ///< Number of splats
  bool isLoaded = false;   ///< Data loaded flag
};

/**
 * @brief OpenGL renderer for 3D Gaussian splats.
 *
 * Manages GPU buffers and rendering pipeline for Gaussian splat data.
 */
class Renderer {
 public:
  Renderer();
  ~Renderer();

  /**
   * @brief Load splat data from DataTable.
   * @param dataTable Input splat data
   */
  void loadSplats(const splat::DataTable& dataTable);

  /**
   * @brief Unload current splat data.
   */
  void unloadSplats();

  /**
   * @brief Render splats with given view-projection matrix.
   * @param vp View-projection matrix
   * @param eyePosition Camera position in world space
   */
  void render(const glm::mat4& vp, const glm::vec3& eyePosition);

  /**
   * @brief Check if splat data is loaded.
   * @return true if data is loaded
   */
  bool hasSplats() const { return splatData.isLoaded; }

  /**
   * @brief Get number of loaded splats.
   * @return Splat count
   */
  size_t getSplatCount() const { return splatData.splatCount; }

  /**
   * @brief Initialize the renderer (shaders, etc).
   */
  void initialize();

  /**
   * @brief Set point size for fallback rendering.
   * @param size Point size in pixels
   */
  void setPointSize(float size) { pointSize = size; }

 private:
  SplatData splatData;
  std::unique_ptr<Shader> shader;
  float pointSize = 3.0f;
  bool initialized = false;

  void createShader();
};

}  // namespace viewer
