#version 430 core

layout(location = 0) in vec2 vertex_offset;

layout(std430, binding = 0) buffer bX { float x[]; };
layout(std430, binding = 1) buffer bY { float y[]; };
layout(std430, binding = 2) buffer bZ { float z[]; };

layout(std430, binding = 3) buffer bS0 { float s0[]; };
layout(std430, binding = 4) buffer bS1 { float s1[]; };
layout(std430, binding = 5) buffer bS2 { float s2[]; };

layout(std430, binding = 6) buffer bR { float r[]; };
layout(std430, binding = 7) buffer bG { float g[]; };
layout(std430, binding = 8) buffer bB { float b[]; };
layout(std430, binding = 9) buffer bA { float a[]; };

layout(std430, binding = 10) buffer bRot0 { float r0[]; };
layout(std430, binding = 11) buffer bRot1 { float r1[]; };
layout(std430, binding = 12) buffer bRot2 { float r2[]; };
layout(std430, binding = 13) buffer bRot3 { float r3[]; };

layout(std430, binding = 14) buffer bIdx { uint sorted_indices[]; };

uniform mat4 view;
uniform mat4 projection;
uniform vec2 focal;
uniform vec2 screen_size;

out vec2 v_offset;
out vec4 v_color;
out vec3 v_conic;

void main() {
  uint id = sorted_indices[gl_InstanceID];

  vec3 center = vec3(x[id], y[id], z[id]);
  vec3 s = exp(vec3(s0[id], s1[id], s2[id]));
  vec4 q = normalize(vec4(r0[id], r1[id], r2[id], r3[id]));  // w,x,y,z

  mat3 R = mat3(1.0 - 2.0 * (q.y * q.y + q.z * q.z), 2.0 * (q.x * q.y - q.w * q.z), 2.0 * (q.x * q.z + q.w * q.y),
                2.0 * (q.x * q.y + q.w * q.z), 1.0 - 2.0 * (q.x * q.x + q.z * q.z), 2.0 * (q.y * q.z - q.w * q.x),
                2.0 * (q.x * q.z - q.w * q.y), 2.0 * (q.y * q.z + q.w * q.x), 1.0 - 2.0 * (q.x * q.x + q.y * q.y));

  mat3 M = R * mat3(s.x, 0, 0, 0, s.y, 0, 0, 0, s.z);
  mat3 Sigma = M * transpose(M);

  vec4 cam_pos = view * vec4(center, 1.0);
  if (cam_pos.z > -0.1) {
    gl_Position = vec4(0.0 / 0.0);
    return;
  }

  mat3 J = mat3(focal.x / cam_pos.z, 0, -(focal.x * cam_pos.x) / (cam_pos.z * cam_pos.z), 0, focal.y / cam_pos.z,
                -(focal.y * cam_pos.y) / (cam_pos.z * cam_pos.z), 0, 0, 0);
  mat3 W = transpose(mat3(view));
  mat3 T = W * J;
  mat3 cov2d = transpose(T) * Sigma * T;
  cov2d[0][0] += 0.3;
  cov2d[1][1] += 0.3;

  float det = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];
  v_conic = vec3(cov2d[1][1] / det, -cov2d[0][1] / det, cov2d[0][0] / det);

  float mid = 0.5 * (cov2d[0][0] + cov2d[1][1]);
  float lambda = mid + sqrt(max(0.1, mid * mid - det));
  float radius = ceil(3.0 * sqrt(lambda));

  v_offset = vertex_offset * radius;
  v_color = vec4(r[id], g[id], b[id], a[id]);

  vec4 p = projection * cam_pos;
  gl_Position = vec4(p.xy + v_offset / screen_size * 2.0 * p.w, p.zw);
}
