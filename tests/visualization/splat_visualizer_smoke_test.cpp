#include <splat/visualization/splat_visualizer.h>

int main() {
  splat::SplatVisualizer visualizer("smoke");
  visualizer.setWindowSize(320, 240);
  visualizer.setAxesEnabled(false);

  splat::SplatRenderOptions options;
  options.sortBackToFront = true;
  options.clampColors = true;

  return visualizer.wasStopped() ? 1 : 0;
}
