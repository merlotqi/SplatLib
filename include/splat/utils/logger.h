/**
 * @file splat/utils/logger.h
 * @brief Lightweight logging to stdout.
 *
 */

#pragma once

#include <cstdarg>
#include <cstdio>
#include <iostream>
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
  logLevel level_ = logLevel::normal;
  std::mutex log_mutex_;

  Logger() = default;

  void logInternal(const char* prefix, const char* file, int line, const char* format, va_list args) {
    if (level_ == logLevel::silent) {
      return;
    }

    va_list args_copy;
    va_copy(args_copy, args);
    int len = std::vsnprintf(nullptr, 0, format, args_copy);
    va_end(args_copy);

    if (len < 0) {
      return;
    }

    std::vector<char> buf(len + 1);
    std::vsnprintf(buf.data(), buf.size(), format, args);

    std::string_view file_sv(file);
    size_t last_slash = file_sv.find_last_of("/\\");
    if (last_slash != std::string_view::npos) {
      file_sv.remove_prefix(last_slash + 1);
    }

    // build the complete line before locking so we can output atomically
    std::string line_buf;
    line_buf.reserve(64 + len);
    line_buf += '[';
    line_buf += prefix;
    line_buf += "] ";
    line_buf += file_sv;
    line_buf += ':';
    line_buf += std::to_string(line);
    line_buf += " > ";
    line_buf.append(buf.data(), len);
    line_buf += '\n';

    std::lock_guard<std::mutex> lock(log_mutex_);
    std::cout << line_buf;
    std::fflush(stdout);
  }

 public:
  static Logger& instance() {
    static Logger instance;
    return instance;
  }
  ~Logger() = default;
  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  void setQuiet(bool quiet) { level_ = quiet ? logLevel::silent : logLevel::normal; }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((format(printf, 4, 5)))
#endif
  void info(const char* file, int line, const char* format, ...) {
    va_list args;
    va_start(args, format);
    logInternal("INFO", file, line, format, args);
    va_end(args);
  }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((format(printf, 4, 5)))
#endif
  void warn(const char* file, int line, const char* format, ...) {
    va_list args;
    va_start(args, format);
    logInternal("WARN", file, line, format, args);
    va_end(args);
  }

#if defined(__GNUC__) || defined(__clang__)
  __attribute__((format(printf, 4, 5)))
#endif
  void error(const char* file, int line, const char* format, ...) {
    va_list args;
    va_start(args, format);
    logInternal("ERROR", file, line, format, args);
    va_end(args);
  }
};

}  // namespace splat

#define LOG_INFO(format, ...) splat::Logger::instance().info(__FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) splat::Logger::instance().warn(__FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) splat::Logger::instance().error(__FILE__, __LINE__, format, ##__VA_ARGS__)
