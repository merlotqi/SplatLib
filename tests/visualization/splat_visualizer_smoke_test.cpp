#include <splat/visualization/splat_visualizer.h>

int main() {
  splat::SplatVisualizer visualizer("smoke");
  visualizer.setWindowSize(320, 240);
  visualizer.setAxesEnabled(false);
  return visualizer.wasStopped() ? 1 : 0;
}
