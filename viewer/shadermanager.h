#pragma once

#include <GL/glew.h>

#include <string>
#include <unordered_map>

class ShaderManager {
 public:
  static ShaderManager& instance();

  GLuint loadShader(const std::string& vertexPath, const std::string& fragmentPath);
  GLuint getShader(const std::string& name);
  void reloadAll();

 private:
  ShaderManager() = default;
  ~ShaderManager();

  struct ShaderProgram {
    GLuint id;
    std::string vertex_path;
    std::string fragment_path;
    time_t vertex_mtime;
    time_t fragment_mtime;
  };

  std::unordered_map<std::string, ShaderProgram> shaders_;

  GLuint compileShader(GLenum type, const std::string& source);
  GLuint linkProgram(GLuint vertex, GLuint fragment);
  std::string loadFile(const std::string& path);
  time_t getFileMTime(const std::string& path);
};
