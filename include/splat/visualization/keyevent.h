#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace splat {

enum class KeyAction : uint8_t {
  Press = 0,
  Release = 1,
};

enum class KeyModifier : uint8_t {
  None = 0,
  Shift = 1u << 0u,
  Control = 1u << 1u,
  Alt = 1u << 2u,
};

constexpr KeyModifier operator|(KeyModifier lhs, KeyModifier rhs) noexcept {
  return static_cast<KeyModifier>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

constexpr KeyModifier operator&(KeyModifier lhs, KeyModifier rhs) noexcept {
  return static_cast<KeyModifier>(static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs));
}

constexpr bool hasModifier(KeyModifier modifiers, KeyModifier flag) noexcept {
  return static_cast<uint8_t>(modifiers & flag) != 0;
}

struct KeyEvent {
  KeyAction action{KeyAction::Press};
  KeyModifier modifiers{KeyModifier::None};
  std::string keySym;
  char keyCode{'\0'};
  int repeatCount{0};

  bool isRepeat() const noexcept { return repeatCount > 0; }
  bool shift() const noexcept { return hasModifier(modifiers, KeyModifier::Shift); }
  bool control() const noexcept { return hasModifier(modifiers, KeyModifier::Control); }
  bool alt() const noexcept { return hasModifier(modifiers, KeyModifier::Alt); }
};

using KeyEventCallback = std::function<void(const KeyEvent&)>;

}  // namespace splat
