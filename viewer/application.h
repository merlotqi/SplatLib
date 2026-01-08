#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <memory>
#include <string>

#include "camera.h"
#include "splatrenderer.h"

class Application {
 public:
  Application(int width = 1280, int height = 720);
  ~Application();

  int run();
  bool loadSplatData(const std::string& filepath);

 private:
  bool initGLFW();
  bool initImGui();
  void mainLoop();
  void cleanup();

  void renderUI();
  void renderMainMenu();
  void renderSettingsPanel();
  void renderStatisticsPanel();
  void renderHelpPanel();

  void update(float deltaTime);
  void render();

  void resetCamera();
  void fitToView();
  void takeScreenshot();

  bool openFileDialog(std::string& outPath);

 private:
  GLFWwindow* window_;
  int width_, height_;

  std::unique_ptr<SplatRenderer> renderer_;
  Camera camera_;

  bool show_settings_ = true;
  bool show_statistics_ = true;
  bool show_help_ = false;
  bool auto_rotate_ = false;
  bool sort_enabled_ = true;

  float focal_length_ = 1000.0f;
  float point_scale_ = 1.0f;
  int sort_interval_ = 10;
  float auto_rotate_speed_ = 0.5f;

  std::string current_file_;
  bool is_loaded_ = false;
  bool is_playing_ = true;
  float frame_time_ = 0.0f;
  int frame_count_ = 0;

  bool mouse_pressed_ = false;
  double last_mouse_x_ = 0.0;
  double last_mouse_y_ = 0.0;
};
