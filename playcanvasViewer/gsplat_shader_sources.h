
#pragma once
namespace playcanvas_viewer {

inline constexpr const char* kGSplatVertexShader = R"GLSL(
#version 330 core
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec3 aCenter;
layout(location = 2) in vec4 aRotation; // w,x,y,z stored as x,y,z,w
layout(location = 3) in vec3 aScale;
layout(location = 4) in vec4 aColor;

uniform mat4 uView;
uniform mat4 uProjection;
uniform vec4 uViewport;
uniform vec4 uCameraParams;
uniform float uMinPixelSize;
uniform float uSizeScale;

out vec2 vUv;
out vec4 vColor;

mat3 quatToMat3(vec4 q) {
    // PlayCanvas: q.x=w, q.y=x, q.z=y, q.w=z
    float w = q.x, x = q.y, y = q.z, z = q.w;
    float x2 = x + x, y2 = y + y, z2 = z + z;
    float xx = x * x2, xy = x * y2, xz = x * z2;
    float yy = y * y2, yz = y * z2, zz = z * z2;
    float wx = w * x2, wy = w * y2, wz = w * z2;
    return mat3(
        1.0 - (yy + zz), xy - wz, xz + wy,
        xy + wz, 1.0 - (xx + zz), yz - wx,
        xz - wy, yz + wx, 1.0 - (xx + yy)
    );
}

void computeCovariance(vec4 rotation, vec3 scale, out vec3 covA, out vec3 covB) {
    mat3 rot = quatToMat3(rotation);
    mat3 M = transpose(mat3(scale.x * rot[0], scale.y * rot[1], scale.z * rot[2]));
    covA = vec3(dot(M[0], M[0]), dot(M[0], M[1]), dot(M[0], M[2]));
    covB = vec3(dot(M[1], M[1]), dot(M[1], M[2]), dot(M[2], M[2]));
}

const float COV_REGULARIZATION = 0.3;
const float VMIN_CLAMP = 1024.0;

bool initCorner(
    vec3 viewCenter, mat3 modelView3, vec4 centerProj, float projMat00,
    vec3 covA, vec3 covB, out vec2 offset, out vec2 uv
) {
    mat3 Vrk = mat3(
        covA.x, covA.y, covA.z,
        covA.y, covB.x, covB.y,
        covA.z, covB.y, covB.z
    );
    float focal = uViewport.x * projMat00;
    vec3 vp = (uCameraParams.w == 1.0) ? vec3(0.0, 0.0, 1.0) : viewCenter;
    float J1 = focal / vp.z;
    vec2 J2 = -J1 / vp.z * vp.xy;
    mat3 J = mat3(J1, 0.0, J2.x, 0.0, J1, J2.y, 0.0, 0.0, 0.0);
    mat3 W = transpose(modelView3);
    mat3 T = W * J;
    mat3 cov = transpose(T) * Vrk * T;
    float diagonal1 = cov[0][0] + COV_REGULARIZATION;
    float offDiagonal = cov[0][1];
    float diagonal2 = cov[1][1] + COV_REGULARIZATION;
    float mid = 0.5 * (diagonal1 + diagonal2);
    float radius = sqrt(0.25 * (diagonal1 - diagonal2) * (diagonal1 - diagonal2) + offDiagonal * offDiagonal);
    float lambda1 = mid + radius;
    float lambda2 = max(mid - radius, 0.1);
    float vmin = min(VMIN_CLAMP, min(uViewport.x, uViewport.y));
    float l1 = 2.0 * min(sqrt(2.0 * lambda1), vmin) * uSizeScale;
    float l2 = 2.0 * min(sqrt(2.0 * lambda2), vmin) * uSizeScale;
    if (max(l1, l2) < uMinPixelSize) return false;
    vec2 c = centerProj.ww * uViewport.zw;
    if (any(greaterThan(abs(centerProj.xy) - vec2(max(l1, l2)) * c, centerProj.ww))) return false;
    vec2 diagonalVectorRaw = vec2(offDiagonal, lambda1 - diagonal1);
    float diagonalVectorLength = length(diagonalVectorRaw);
    vec2 diagonalVector = diagonalVectorLength > 1e-6 ? diagonalVectorRaw / diagonalVectorLength : vec2(1.0, 0.0);
    vec2 v1 = l1 * diagonalVector;
    vec2 v2 = l2 * vec2(diagonalVector.y, -diagonalVector.x);
    offset = (aCorner.x * v1 + aCorner.y * v2) * c;
    uv = aCorner;
    return true;
}

void main() {
    vec4 centerView4 = uView * vec4(aCenter, 1.0);
    if (uCameraParams.w == 0.0 && centerView4.z > 0.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        vUv = vec2(2.0);
        vColor = vec4(0.0);
        return;
    }
    vec4 centerProj = uProjection * centerView4;
    centerProj.z = clamp(centerProj.z, -abs(centerProj.w), abs(centerProj.w));
    mat3 modelView3 = mat3(uView);
    vec3 covA, covB;
    computeCovariance(aRotation, aScale, covA, covB);
    vec2 offset, uv;
    if (!initCorner(centerView4.xyz, modelView3, centerProj, uProjection[0][0], covA, covB, offset, uv)) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        vUv = vec2(2.0);
        vColor = vec4(0.0);
        return;
    }
    gl_Position = centerProj + vec4(offset, 0.0, 0.0);
    vUv = uv;
    vColor = aColor;
}
)GLSL";

inline constexpr const char* kGSplatFragmentShader = R"GLSL(
#version 330 core
in vec2 vUv;
in vec4 vColor;
uniform float uGlobalOpacity;
uniform float uAlphaDiscardThreshold;
out vec4 fragColor;

const float EXP4 = exp(-4.0);
const float INV_EXP4 = 1.0 / (1.0 - EXP4);
float normExp(float x) { return (exp(x * -4.0) - EXP4) * INV_EXP4; }

void main() {
    float A = dot(vUv, vUv);
    if (A > 1.0) discard;
    float alpha = normExp(A) * clamp(vColor.a * uGlobalOpacity, 0.0, 1.0);
    if (alpha < uAlphaDiscardThreshold) discard;
    fragColor = vec4(max(vColor.rgb, vec3(0.0)) * alpha, alpha);
}
)GLSL";

} // namespace playcanvas_viewer
