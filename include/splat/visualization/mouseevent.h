#pragma once

#include <splat/visualization/keyevent.h>

#include <cstdint>
#include <functional>

namespace splat {

enum class MouseAction : uint8_t {
  Move = 0,
  Press = 1,
  Release = 2,
  DoubleClick = 3,
  Wheel = 4,
};

enum class MouseButton : uint8_t {
  None = 0,
  Left = 1,
  Middle = 2,
  Right = 3,
  Button4 = 4,
  Button5 = 5,
};

struct MouseEvent {
  MouseAction action{MouseAction::Move};
  MouseButton button{MouseButton::None};
  KeyModifier modifiers{KeyModifier::None};
  int x{0};
  int y{0};
  int lastX{0};
  int lastY{0};
  int wheelDelta{0};
  int repeatCount{0};

  int dx() const noexcept { return x - lastX; }
  int dy() const noexcept { return y - lastY; }
  bool shift() const noexcept { return hasModifier(modifiers, KeyModifier::Shift); }
  bool control() const noexcept { return hasModifier(modifiers, KeyModifier::Control); }
  bool alt() const noexcept { return hasModifier(modifiers, KeyModifier::Alt); }
};

using MouseEventCallback = std::function<void(const MouseEvent&)>;

}  // namespace splat
