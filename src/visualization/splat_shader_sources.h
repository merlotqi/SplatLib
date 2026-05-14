#pragma once

namespace splat::visualization {

inline constexpr const char* kGaussianVertexShaderSource = R"(
#version 330 core

layout (location = 0) in vec2 aCorner;
layout (location = 1) in vec3 aPosition;
layout (location = 2) in vec3 aShDc;
layout (location = 3) in vec3 aLogScale;
layout (location = 4) in float aOpacity;
layout (location = 5) in vec4 aRotation;

uniform mat4 uModelView;
uniform mat4 uModelClip;
uniform vec2 uViewportSize;
uniform float uParallelProjection;
uniform float uProjectionScaleX;
uniform float uProjectionScaleY;
uniform float uSizeScale;
uniform float uMinPointSize;
uniform float uMaxPointSize;
uniform float uClampColors;
uniform float uAlphaDiscardThreshold;

out vec2 vGaussianUv;
out vec4 vColor;

mat3 quaternionToMatrix(vec4 q) {
    vec4 normalized = normalize(q);
    float w = normalized.x;
    float x = normalized.y;
    float y = normalized.z;
    float z = normalized.w;

    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;
    float wx = w * x;
    float wy = w * y;
    float wz = w * z;

    return mat3(
        1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz),       2.0 * (xz - wy),
        2.0 * (xy - wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx),
        2.0 * (xz + wy),       2.0 * (yz - wx),       1.0 - 2.0 * (xx + yy)
    );
}

void main() {
    vec4 cameraPos4 = uModelView * vec4(aPosition, 1.0);
    vec3 cameraPos = cameraPos4.xyz;

    vGaussianUv = vec2(0.0);
    vColor = vec4(0.0);

    if (cameraPos.z >= -1e-4) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }

    vec3 scale = exp(aLogScale);
    mat3 rotation = quaternionToMatrix(aRotation);
    float opacity = 1.0 / (1.0 + exp(-aOpacity));

    float fx = uProjectionScaleX * uViewportSize.x;
    float fy = uProjectionScaleY * uViewportSize.y;
    mat3 jacobian;

    if (uParallelProjection > 0.5) {
        jacobian = mat3(
            fx, 0.0, 0.0,
            0.0, fy, 0.0,
            0.0, 0.0, 0.0
        );
    } else {
        float z = min(cameraPos.z, -1e-4);
        float jx = fx / z;
        float jy = fy / z;
        jacobian = mat3(
            jx, 0.0, -jx / z * cameraPos.x,
            0.0, jy, -jy / z * cameraPos.y,
            0.0, 0.0, 0.0
        );
    }

    mat3 scaledRotation = transpose(mat3(
        scale.x * rotation[0],
        scale.y * rotation[1],
        scale.z * rotation[2]
    ));
    mat3 covarianceModel = transpose(scaledRotation) * scaledRotation;
    mat3 viewJacobian = transpose(mat3(uModelView)) * jacobian;
    mat3 covarianceScreen = transpose(viewJacobian) * covarianceModel * viewJacobian;

    float cov00 = covarianceScreen[0][0];
    float cov01 = covarianceScreen[0][1];
    float cov11 = covarianceScreen[1][1];

    cov00 += 0.3;
    cov11 += 0.3;

    float trace = cov00 + cov11;
    float discriminant = sqrt(max((cov00 - cov11) * (cov00 - cov11) + 4.0 * cov01 * cov01, 0.0));
    float lambda1 = max(0.5 * (trace + discriminant), 1e-6);
    float lambda2 = max(0.5 * (trace - discriminant), 0.1);
    float diagonalSize = uSizeScale * 2.0 * sqrt(2.0 * lambda1);
    float sideSize = uSizeScale * 2.0 * sqrt(2.0 * lambda2);
    if (max(diagonalSize, sideSize) < uMinPointSize || opacity <= uAlphaDiscardThreshold) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    diagonalSize = min(diagonalSize, uMaxPointSize);
    sideSize = min(sideSize, uMaxPointSize);

    vec2 diagonalVector = vec2(cov01, lambda1 - cov00);
    if (dot(diagonalVector, diagonalVector) < 1e-6) {
        diagonalVector = vec2(1.0, 0.0);
    } else {
        diagonalVector = normalize(diagonalVector);
    }
    vec2 v1 = diagonalSize * diagonalVector;
    vec2 v2 = sideSize * vec2(diagonalVector.y, -diagonalVector.x);

    vec4 centerClip = uModelClip * vec4(aPosition, 1.0);
    centerClip.z = clamp(centerClip.z, -abs(centerClip.w), abs(centerClip.w));
    vec2 clipScale = centerClip.ww / uViewportSize;
    float alphaClip = max(uAlphaDiscardThreshold, 1e-6);
    float cornerClip = min(1.0, sqrt(max(0.0, log(opacity / alphaClip))) * 0.5);
    vec2 clippedCorner = aCorner * cornerClip;
    gl_Position = centerClip + vec4((clippedCorner.x * v1 + clippedCorner.y * v2) * clipScale, 0.0, 0.0);
    vGaussianUv = clippedCorner;

    const float SH_C0 = 0.28209479177387814;
    vec3 decodedColor = aShDc * SH_C0 + vec3(0.5);
    vColor = vec4(uClampColors > 0.5 ? clamp(decodedColor, 0.0, 1.0) : decodedColor, opacity);
}
)";

inline constexpr const char* kGaussianFragmentShaderSource = R"(
#version 330 core

in vec2 vGaussianUv;
in vec4 vColor;

uniform float uGlobalOpacity;
uniform float uAlphaDiscardThreshold;

out vec4 fragColor;

const float EXP4 = exp(-4.0);
const float INV_EXP4 = 1.0 / (1.0 - EXP4);

void main() {
    float A = dot(vGaussianUv, vGaussianUv);
    if (A > 1.0) {
        discard;
    }

    float alpha = ((exp(A * -4.0) - EXP4) * INV_EXP4) * vColor.a * uGlobalOpacity;
    if (alpha < uAlphaDiscardThreshold) {
        discard;
    }

    fragColor = vec4(vColor.rgb * alpha, alpha);
}
)";

}  // namespace splat::visualization
