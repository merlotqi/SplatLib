#pragma once

namespace splat::visualization {

inline constexpr const char* kGaussianVertexShaderSource = R"(
#version 330 core

layout (location = 0) in vec3 aPosition;
layout (location = 1) in vec3 aShDc;
layout (location = 2) in vec3 aLogScale;
layout (location = 3) in float aOpacity;
layout (location = 4) in vec4 aRotation;

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

flat out vec3 vColor;
flat out float vOpacity;
flat out vec3 vConic;
flat out float vPointSize;

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

    vColor = vec3(0.0);
    vOpacity = 0.0;
    vConic = vec3(1.0, 0.0, 1.0);
    vPointSize = 0.0;

    if (cameraPos.z >= -1e-4) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        return;
    }

    vec3 scale = exp(aLogScale);
    mat3 rotation = quaternionToMatrix(aRotation);
    mat3 scaleMatrix = mat3(
        vec3(scale.x, 0.0, 0.0),
        vec3(0.0, scale.y, 0.0),
        vec3(0.0, 0.0, scale.z)
    );
    mat3 basisCam = mat3(uModelView) * rotation * scaleMatrix;
    mat3 covarianceCam = basisCam * transpose(basisCam);

    float fx = uProjectionScaleX * 0.5 * uViewportSize.x;
    float fy = uProjectionScaleY * 0.5 * uViewportSize.y;
    vec3 jx;
    vec3 jy;

    if (uParallelProjection > 0.5) {
        jx = vec3(fx, 0.0, 0.0);
        jy = vec3(0.0, fy, 0.0);
    } else {
        float invNegZ = 1.0 / max(-cameraPos.z, 1e-4);
        jx = vec3(fx * invNegZ, 0.0, fx * cameraPos.x * invNegZ * invNegZ);
        jy = vec3(0.0, fy * invNegZ, fy * cameraPos.y * invNegZ * invNegZ);
    }

    float cov00 = dot(jx, covarianceCam * jx);
    float cov01 = dot(jx, covarianceCam * jy);
    float cov11 = dot(jy, covarianceCam * jy);

    cov00 += 0.3;
    cov11 += 0.3;

    float determinant = max(cov00 * cov11 - cov01 * cov01, 1e-6);
    vConic = vec3(cov11 / determinant, -cov01 / determinant, cov00 / determinant);

    float trace = cov00 + cov11;
    float discriminant = sqrt(max((cov00 - cov11) * (cov00 - cov11) + 4.0 * cov01 * cov01, 0.0));
    float maxEigenvalue = max(0.5 * (trace + discriminant), 1e-6);
    vPointSize = clamp(uSizeScale * sqrt(maxEigenvalue) * 2.0, uMinPointSize, uMaxPointSize);

    gl_Position = uModelClip * vec4(aPosition, 1.0);
    gl_PointSize = vPointSize;

    const float SH_C0 = 0.28209479177387814;
    vec3 decodedColor = aShDc * SH_C0 + vec3(0.5);
    vColor = uClampColors > 0.5 ? clamp(decodedColor, 0.0, 1.0) : decodedColor;
    vOpacity = 1.0 / (1.0 + exp(-aOpacity));
}
)";

inline constexpr const char* kGaussianFragmentShaderSource = R"(
#version 330 core

flat in vec3 vColor;
flat in float vOpacity;
flat in vec3 vConic;
flat in float vPointSize;

uniform float uGlobalOpacity;
uniform float uAlphaDiscardThreshold;

out vec4 fragColor;

void main() {
    vec2 pixelOffset = (gl_PointCoord - vec2(0.5)) * vPointSize;
    float quadForm = pixelOffset.x * (vConic.x * pixelOffset.x + vConic.y * pixelOffset.y) +
                     pixelOffset.y * (vConic.y * pixelOffset.x + vConic.z * pixelOffset.y);
    quadForm = max(quadForm, 0.0);
    float alpha = exp(-0.5 * quadForm) * vOpacity * uGlobalOpacity;
    if (alpha < uAlphaDiscardThreshold) {
        discard;
    }

    fragColor = vec4(vColor, alpha);
}
)";

}  // namespace splat::visualization
