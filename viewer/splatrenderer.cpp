#include "splatrenderer.h"
#include "shadermanager.h"
#include <iostream>
#include <Eigen/Dense>

SplatRenderer::SplatRenderer() {
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    
    float quad[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

SplatRenderer::~SplatRenderer() {
    unload();
    
    if (vao_) glDeleteVertexArrays(1, &vao_);
    if (vbo_) glDeleteBuffers(1, &vbo_);
    if (program_) glDeleteProgram(program_);
}

bool SplatRenderer::loadFromDataTable(const splat::DataTable& table) {
    unload();
    
    point_count_ = table.getNumRows();
    if (point_count_ == 0) {
        std::cerr << "No data in table" << std::endl;
        return false;
    }
    
    data_table_ = table.clone();
    
    program_ = ShaderManager::instance().loadShader("shaders/splat.vert", "shaders/splat.frag");
    if (program_ == 0) {
        std::cerr << "Failed to load shaders" << std::endl;
        return false;
    }
    
    view_loc_ = glGetUniformLocation(program_, "view");
    proj_loc_ = glGetUniformLocation(program_, "projection");
    focal_loc_ = glGetUniformLocation(program_, "focal");
    screen_size_loc_ = glGetUniformLocation(program_, "screen_size");
    point_scale_loc_ = glGetUniformLocation(program_, "point_scale");
    
    if (!createBuffers(*data_table_)) {
        std::cerr << "Failed to create buffers" << std::endl;
        return false;
    }
    
    sorted_indices_.resize(point_count_);
    for (uint32_t i = 0; i < point_count_; ++i) {
        sorted_indices_[i] = i;
    }
    
    if (!ssbos_.empty()) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbos_.back());
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       sorted_indices_.size() * sizeof(uint32_t),
                       sorted_indices_.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
    
    is_loaded_ = true;
    std::cout << "Loaded " << point_count_ << " Gaussians" << std::endl;
    
    return true;
}

void SplatRenderer::unload() {
    if (!ssbos_.empty()) {
        glDeleteBuffers(static_cast<GLsizei>(ssbos_.size()), ssbos_.data());
        ssbos_.clear();
    }
    
    data_table_.reset();
    sorted_indices_.clear();
    point_count_ = 0;
    is_loaded_ = false;
}

bool SplatRenderer::createBuffers(const splat::DataTable& table) {
    std::vector<std::string> columns = {
        "x", "y", "z",
        "scale_0", "scale_1", "scale_2",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "rot_0", "rot_1", "rot_2", "rot_3"
    };
    
    for (const auto& col : columns) {
        if (!table.hasColumn(col)) {
            std::cerr << "Missing required column: " << col << std::endl;
            return false;
        }
    }
    
    ssbos_.resize(columns.size() + 1); // +1 for indices
    glGenBuffers(ssbos_.size(), ssbos_.data());
    
    for (size_t i = 0; i < columns.size(); ++i) {
        const auto& col = table.getColumnByName(columns[i]).asVector<float>();
        
        if (col.size() != point_count_) {
            std::cerr << "Column " << columns[i] << " size mismatch: " 
                      << col.size() << " != " << point_count_ << std::endl;
            return false;
        }
        
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbos_[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER,
                    col.size() * sizeof(float),
                    col.data(),
                    GL_STATIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, ssbos_[i]);
        
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cerr << "OpenGL error creating SSBO for " << columns[i] 
                      << ": " << err << std::endl;
            return false;
        }
    }
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbos_.back());
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                sorted_indices_.size() * sizeof(uint32_t),
                sorted_indices_.data(),
                GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, ssbos_.back());
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    
    GLint maxBindings;
    glGetIntegerv(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS, &maxBindings);
    std::cout << "Max SSBO bindings: " << maxBindings << std::endl;
    
    return true;
}

void SplatRenderer::render(const glm::mat4& view, const glm::mat4& projection,
                          float focal_length,
                          const glm::vec2& screen_size) {
    if (!is_loaded_ || !program_) return;
    
    glUseProgram(program_);
    
    if (view_loc_ != -1) glUniformMatrix4fv(view_loc_, 1, GL_FALSE, &view[0][0]);
    if (proj_loc_ != -1) glUniformMatrix4fv(proj_loc_, 1, GL_FALSE, &projection[0][0]);
    if (focal_loc_ != -1) glUniform2f(focal_loc_, focal_length, focal_length);
    if (screen_size_loc_ != -1) glUniform2f(screen_size_loc_, screen_size.x, screen_size.y);
    if (point_scale_loc_ != -1) glUniform1f(point_scale_loc_, point_scale_);
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);
    glDisable(GL_DEPTH_TEST);
    
    for (size_t i = 0; i < ssbos_.size(); ++i) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, ssbos_[i]);
    }
    
    glBindVertexArray(vao_);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, point_count_);
    glBindVertexArray(0);
    
    for (size_t i = 0; i < ssbos_.size(); ++i) {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0);
    }
    
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
}

void SplatRenderer::updateSorting(const glm::mat4& view) {
    if (!is_loaded_ || !sort_enabled_ || ssbos_.empty()) return;
    
    static int frame_counter = 0;
    frame_counter++;
    if (frame_counter < sort_interval_) return;
    frame_counter = 0;
    
    Eigen::Matrix4f eigen_view;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            eigen_view(i, j) = view[j][i];
        }
    }
    
    try {
        //splat::generateOrdering(data_table_.get(), absl::MakeSpan(sorted_indices_));
        
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbos_.back());
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        0,
                        sorted_indices_.size() * sizeof(uint32_t),
                        sorted_indices_.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        
    } catch (const std::exception& e) {
        std::cerr << "Sorting error: " << e.what() << std::endl;
    }
}

size_t SplatRenderer::getMemoryUsage() const {
    if (!is_loaded_) return 0;
    
    size_t cpu_memory = 0;
    
    std::vector<std::string> columns = {
        "x", "y", "z", "scale_0", "scale_1", "scale_2",
        "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "rot_0", "rot_1", "rot_2", "rot_3"
    };
    
    for (const auto& col : columns) {
        if (data_table_->hasColumn(col)) {
            const auto& vec = data_table_->getColumnByName(col).asVector<float>();
            cpu_memory += vec.size() * sizeof(float);
        }
    }
    
    cpu_memory += sorted_indices_.size() * sizeof(uint32_t);
    
    size_t gpu_memory = cpu_memory;
    
    return cpu_memory + gpu_memory;
}

void SplatRenderer::reloadShaders() {
    if (program_) {
        glDeleteProgram(program_);
        program_ = 0;
    }
    
    program_ = ShaderManager::instance().loadShader("shaders/splat.vert", "shaders/splat.frag");
    
    if (program_ != 0) {
        view_loc_ = glGetUniformLocation(program_, "view");
        proj_loc_ = glGetUniformLocation(program_, "projection");
        focal_loc_ = glGetUniformLocation(program_, "focal");
        screen_size_loc_ = glGetUniformLocation(program_, "screen_size");
        point_scale_loc_ = glGetUniformLocation(program_, "point_scale");
    }
}
