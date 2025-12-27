/**
 * splat - A C++ library for reading and writing 3D Gaussian Splatting (splat) files.
 *
 * This library provides functionality to convert, manipulate, and process
 * 3D Gaussian splatting data formats used in real-time neural rendering.
 *
 * This file is part of splat.
 *
 * splat is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * splat is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * For more information, visit the project's homepage or contact the author.
 */

#pragma once

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace splat {

enum class logLevel {
  silent,
  normal
};

class Logger {
  logLevel level = logLevel::normal;
  std::mutex log_mutex;

  Logger() = default;

  void log_internal(const char* prefix, const char* file, int line, const char* format, va_list args) {
    if (level == logLevel::silent) return;

    va_list args_copy;
    va_copy(args_copy, args);
    int len = std::vsnprintf(nullptr, 0, format, args_copy);
    va_end(args_copy);

    if (len < 0) return;

    std::vector<char> buf(len + 1);
    std::vsnprintf(buf.data(), buf.size(), format, args);
    std::string formatted_msg(buf.data());

    std::string_view file_sv(file);
    size_t last_slash = file_sv.find_last_of("/\\");
    if (last_slash != std::string_view::npos) {
      file_sv.remove_prefix(last_slash + 1);
    }

    {
      std::lock_guard<std::mutex> lock(log_mutex);
      std::printf("[%s] %.*s:%d > %s\n", prefix, static_cast<int>(file_sv.size()), file_sv.data(), line,
                  formatted_msg.c_str());
      std::fflush(stdout);
    }
  }

 public:
  static Logger& instance() {
    static Logger instance;
    return instance;
  }

  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  void setQuiet(bool quiet) { level = quiet ? logLevel::silent : logLevel::normal; }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((format(printf, 4, 5)))
#endif
  void info(const char* file, int line, const char* format, ...) {
    va_list args;
    va_start(args, format);
    log_internal("INFO", file, line, format, args);
    va_end(args);
  }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((format(printf, 4, 5)))
#endif
  void warn(const char* file, int line, const char* format, ...) {
    va_list args;
    va_start(args, format);
    log_internal("WARN", file, line, format, args);
    va_end(args);
  }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((format(printf, 4, 5)))
#endif
  void error(const char* file, int line, const char* format, ...) {
    va_list args;
    va_start(args, format);
    log_internal("ERROR", file, line, format, args);
    va_end(args);
  }
};

}  // namespace splat

#define LOG_INFO(format, ...) splat::Logger::instance().info(__FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) splat::Logger::instance().warn(__FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) splat::Logger::instance().error(__FILE__, __LINE__, format, ##__VA_ARGS__)
