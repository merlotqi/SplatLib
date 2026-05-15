#include "gsplat_gl_shader_sources.h"

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_quaternion_matrix_matches_playcanvas_column_major_formula() {
    const std::string_view shader(splat::visualization::kGSplatGLVertexShader);

    require(shader.find("vec4 R2 = R + R;") != std::string_view::npos,
            "quatToMat3 should use the PlayCanvas reference formula");
    require(shader.find("float X = R2.x * R.w;") != std::string_view::npos,
            "quatToMat3 should treat rot_0..3 as PlayCanvas wxyz data");
    require(shader.find("Y.z + X") != std::string_view::npos,
            "quatToMat3 should be written in GLSL column-major order");
    require(shader.find("xy - wz") == std::string_view::npos,
            "quatToMat3 should not use the old transposed row-major formula");
}

} // namespace

int main() {
    test_quaternion_matrix_matches_playcanvas_column_major_formula();
    std::cout << "All PlaycanvasViewerShader tests passed.\n";
    return 0;
}
