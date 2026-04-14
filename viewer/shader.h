/**
 * @file shader.h
 * @brief OpenGL shader compilation and program management.
 */

#pragma once

#include <GL/glew.h>

#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace viewer {

/**
 * @brief OpenGL shader program wrapper.
 *
 * Handles shader source loading, compilation, linking, and
 * uniform variable setting.
 */
class Shader {
 public:
  GLuint id;  ///< Program object ID

  /**
   * @brief Construct shader program from source strings.
   *
   * @param vertexSource Vertex shader GLSL source
   * @param fragmentSource Fragment shader GLSL source
   */
  Shader(const char* vertexSource, const char* fragmentSource) {
    // Compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, nullptr);
    glCompileShader(vertexShader);
    checkCompileErrors(vertexShader, "VERTEX");

    // Compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, nullptr);
    glCompileShader(fragmentShader);
    checkCompileErrors(fragmentShader, "FRAGMENT");

    // Link program
    id = glCreateProgram();
    glAttachShader(id, vertexShader);
    glAttachShader(id, fragmentShader);
    glLinkProgram(id);
    checkCompileErrors(id, "PROGRAM");

    // Clean up shaders
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
  }

  /**
   * @brief Load shader sources from files.
   *
   * @param vertexPath Path to vertex shader file
   * @param fragmentPath Path to fragment shader file
   */
  static Shader fromFiles(const std::filesystem::path& vertexPath, const std::filesystem::path& fragmentPath) {
    std::string vertexCode = readFile(vertexPath);
    std::string fragmentCode = readFile(fragmentPath);
    return Shader(vertexCode.c_str(), fragmentCode.c_str());
  }

  /**
   * @brief Activate shader program.
   */
  void use() const { glUseProgram(id); }

  // Uniform setters
  void setBool(const std::string& name, bool value) const {
    glUniform1i(glGetUniformLocation(id, name.c_str()), static_cast<int>(value));
  }
  void setInt(const std::string& name, int value) const { glUniform1i(glGetUniformLocation(id, name.c_str()), value); }
  void setFloat(const std::string& name, float value) const {
    glUniform1f(glGetUniformLocation(id, name.c_str()), value);
  }
  void setVec3(const std::string& name, const glm::vec3& value) const {
    glUniform3fv(glGetUniformLocation(id, name.c_str()), 1, glm::value_ptr(value));
  }
  void setMat4(const std::string& name, const glm::mat4& value) const {
    glUniformMatrix4fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, glm::value_ptr(value));
  }

 private:
  /**
   * @brief Read file contents as string.
   */
  static std::string readFile(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error(std::string("Failed to open shader file: ") + path.u8string());
    }
    std::stringstream stream;
    stream << file.rdbuf();
    return stream.str();
  }

  /**
   * @brief Check compilation/linking errors.
   */
  static void checkCompileErrors(GLuint shader, const std::string& type) {
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM") {
      glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
      if (!success) {
        glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
        std::cerr << "Shader compilation error (" << type << "): " << infoLog << std::endl;
      }
    } else {
      glGetProgramiv(shader, GL_LINK_STATUS, &success);
      if (!success) {
        glGetProgramInfoLog(shader, 1024, nullptr, infoLog);
        std::cerr << "Program linking error: " << infoLog << std::endl;
      }
    }
  }
};

}  // namespace viewer
