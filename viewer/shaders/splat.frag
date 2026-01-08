#version 430 core
in vec2 v_offset;
in vec4 v_color;
in vec3 v_conic;
out vec4 fragColor;

void main() {
  float d2 = v_conic.x * v_offset.x * v_offset.x + 2.0 * v_conic.y * v_offset.x * v_offset.y +
             v_conic.z * v_offset.y * v_offset.y;

  float power = -0.5 * d2;
  if (power > 0.0) discard;

  float alpha = v_color.a * exp(power);
  if (alpha < 0.005) discard;

  fragColor = vec4(v_color.rgb * alpha, alpha);
}
