#pragma once

#include <GL/glew.h>
#include <splat/splat.h>

#include <glm/glm.hpp>
#include <memory>
#include <vector>

class SplatRenderer {
 public:
  SplatRenderer();
  ~SplatRenderer();

  bool loadFromDataTable(const splat::DataTable& table);
  void unload();

  void render(const glm::mat4& view, const glm::mat4& projection, float focal_length = 1000.0f,
              const glm::vec2& screen_size = glm::vec2(800, 600));

  void updateSorting(const glm::mat4& view);
  void reloadShaders();

  bool isLoaded() const { return is_loaded_; }
  int getPointCount() const { return point_count_; }
  size_t getMemoryUsage() const;

  void setSortEnabled(bool enabled) { sort_enabled_ = enabled; }
  void setSortInterval(int interval) { sort_interval_ = interval; }
  void setPointScale(float scale) { point_scale_ = scale; }

 private:
  bool createBuffers(const splat::DataTable& table);

 private:
  GLuint program_ = 0;
  GLuint vao_ = 0, vbo_ = 0;
  std::vector<GLuint> ssbos_;

  // Uniform locations
  GLint view_loc_ = -1;
  GLint proj_loc_ = -1;
  GLint focal_loc_ = -1;
  GLint screen_size_loc_ = -1;
  GLint point_scale_loc_ = -1;

  std::unique_ptr<splat::DataTable> data_table_;
  std::vector<uint32_t> sorted_indices_;

  int point_count_ = 0;

  bool is_loaded_ = false;
  bool sort_enabled_ = true;
  int sort_interval_ = 10;
  float point_scale_ = 1.0f;
};
