#include "shadermanager.h"

#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <sstream>

ShaderManager& ShaderManager::instance() {
  static ShaderManager instance;
  return instance;
}

ShaderManager::~ShaderManager() {
  for (auto& pair : shaders_) {
    glDeleteProgram(pair.second.id);
  }
}

GLuint ShaderManager::loadShader(const std::string& vertexPath, const std::string& fragmentPath) {
  std::string key = vertexPath + "|" + fragmentPath;

  auto it = shaders_.find(key);
  if (it != shaders_.end()) {
    return it->second.id;
  }

  std::string vertexSource = loadFile(vertexPath);
  std::string fragmentSource = loadFile(fragmentPath);

  if (vertexSource.empty() || fragmentSource.empty()) {
    std::cerr << "Failed to load shader files: " << vertexPath << ", " << fragmentPath << std::endl;
    return 0;
  }

  GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
  GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

  if (!vertexShader || !fragmentShader) {
    if (vertexShader) glDeleteShader(vertexShader);
    if (fragmentShader) glDeleteShader(fragmentShader);
    return 0;
  }

  GLuint program = linkProgram(vertexShader, fragmentShader);

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  if (!program) {
    return 0;
  }

  ShaderProgram shaderProg;
  shaderProg.id = program;
  shaderProg.vertex_path = vertexPath;
  shaderProg.fragment_path = fragmentPath;
  shaderProg.vertex_mtime = getFileMTime(vertexPath);
  shaderProg.fragment_mtime = getFileMTime(fragmentPath);

  shaders_[key] = shaderProg;

  return program;
}

GLuint ShaderManager::getShader(const std::string& name) {
  auto it = shaders_.find(name);
  return (it != shaders_.end()) ? it->second.id : 0;
}

void ShaderManager::reloadAll() {
  for (auto& pair : shaders_) {
    ShaderProgram& prog = pair.second;

    time_t vert_mtime = getFileMTime(prog.vertex_path);
    time_t frag_mtime = getFileMTime(prog.fragment_path);

    if (vert_mtime > prog.vertex_mtime || frag_mtime > prog.fragment_mtime) {
      std::cout << "Reloading shader: " << pair.first << std::endl;

      std::string vert_source = loadFile(prog.vertex_path);
      std::string frag_source = loadFile(prog.fragment_path);

      if (!vert_source.empty() && !frag_source.empty()) {
        GLuint vert_shader = compileShader(GL_VERTEX_SHADER, vert_source);
        GLuint frag_shader = compileShader(GL_FRAGMENT_SHADER, frag_source);

        if (vert_shader && frag_shader) {
          GLuint new_prog = linkProgram(vert_shader, frag_shader);

          if (new_prog) {
            glDeleteProgram(prog.id);
            prog.id = new_prog;
            prog.vertex_mtime = vert_mtime;
            prog.fragment_mtime = frag_mtime;
          }

          glDeleteShader(vert_shader);
          glDeleteShader(frag_shader);
        }
      }
    }
  }
}

GLuint ShaderManager::compileShader(GLenum type, const std::string& source) {
  GLuint shader = glCreateShader(type);
  const char* source_cstr = source.c_str();
  glShaderSource(shader, 1, &source_cstr, nullptr);
  glCompileShader(shader);

  GLint success;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    char info_log[512];
    glGetShaderInfoLog(shader, 512, nullptr, info_log);
    std::cerr << "Shader compilation error:\n" << info_log << std::endl;
    glDeleteShader(shader);
    return 0;
  }

  return shader;
}

GLuint ShaderManager::linkProgram(GLuint vertex, GLuint fragment) {
  GLuint program = glCreateProgram();
  glAttachShader(program, vertex);
  glAttachShader(program, fragment);
  glLinkProgram(program);

  GLint success;
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    char info_log[512];
    glGetProgramInfoLog(program, 512, nullptr, info_log);
    std::cerr << "Program linking error:\n" << info_log << std::endl;
    glDeleteProgram(program);
    return 0;
  }

  return program;
}

std::string ShaderManager::loadFile(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::string exe_dir = "./";
    std::string full_path = exe_dir + path;
    file.open(full_path);

    if (!file.is_open()) {
      std::cerr << "Failed to open file: " << path << std::endl;
      return "";
    }
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

time_t ShaderManager::getFileMTime(const std::string& path) {
  struct stat file_stat;
  if (stat(path.c_str(), &file_stat) == 0) {
    return file_stat.st_mtime;
  }
  return 0;
}
