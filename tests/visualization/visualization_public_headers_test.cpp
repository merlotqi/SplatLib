#include <splat/visualization/splat_visualizer.h>
#include <splat/visualization/keyevent.h>
#include <splat/visualization/mouseevent.h>

#include <type_traits>

#if __has_include(<splat/visualization/gsplat_data.h>)
#define SPLAT_HAS_PUBLIC_GSPLAT_DATA_HEADER 1
#else
#define SPLAT_HAS_PUBLIC_GSPLAT_DATA_HEADER 0
#endif

#if __has_include(<splat/visualization/gsplat_gl_renderer.h>)
#define SPLAT_HAS_PUBLIC_GSPLAT_GL_RENDERER_HEADER 1
#else
#define SPLAT_HAS_PUBLIC_GSPLAT_GL_RENDERER_HEADER 0
#endif

#if __has_include(<splat/visualization/splat_gaussian_prop.h>)
#define SPLAT_HAS_PUBLIC_SPLAT_GAUSSIAN_PROP_HEADER 1
#else
#define SPLAT_HAS_PUBLIC_SPLAT_GAUSSIAN_PROP_HEADER 0
#endif

#if __has_include(<splat/visualization/keyevent.h>)
#define SPLAT_HAS_PUBLIC_KEYEVENT_HEADER 1
#else
#define SPLAT_HAS_PUBLIC_KEYEVENT_HEADER 0
#endif

#if __has_include(<splat/visualization/mouseevent.h>)
#define SPLAT_HAS_PUBLIC_MOUSEEVENT_HEADER 1
#else
#define SPLAT_HAS_PUBLIC_MOUSEEVENT_HEADER 0
#endif

static_assert(SPLAT_HAS_PUBLIC_GSPLAT_DATA_HEADER == 0, "GSplatData is an internal visualization detail");
static_assert(SPLAT_HAS_PUBLIC_GSPLAT_GL_RENDERER_HEADER == 0,
              "GSplatGLRenderer is an internal visualization detail");
static_assert(SPLAT_HAS_PUBLIC_SPLAT_GAUSSIAN_PROP_HEADER == 0,
              "SplatGaussianProp is an internal visualization detail");
static_assert(SPLAT_HAS_PUBLIC_KEYEVENT_HEADER == 1, "KeyEvent is part of the public interaction API");
static_assert(SPLAT_HAS_PUBLIC_MOUSEEVENT_HEADER == 1, "MouseEvent is part of the public interaction API");

int main() {
  static_assert(std::is_class_v<splat::SplatVisualizer>);
  static_assert(std::is_same_v<splat::SplatVisualizer::KeyCallbackHandle, std::size_t>);
  static_assert(std::is_same_v<splat::SplatVisualizer::MouseCallbackHandle, std::size_t>);
  static_assert(std::is_enum_v<splat::KeyAction>);
  static_assert(std::is_enum_v<splat::MouseAction>);
  return 0;
}
