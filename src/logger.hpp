#pragma once
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>

namespace hl {

class Logger {
public:
  // Initialize the logger with a name
  static void init(const std::string &logger_name = "gouda_logger") {
    // Create a console logger
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

    // Create the logger with console sink
    auto logger = std::make_shared<spdlog::logger>(logger_name, console_sink);

    // Set the default log level
    logger->set_level(spdlog::level::info);

    // Set the logger as the default
    spdlog::set_default_logger(logger);

    // Optional: Set pattern
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
  }

  // Set the global log level
  static void set_level(spdlog::level::level_enum level) {
    spdlog::set_level(level);
  }

  // Convenience methods for logging
  template <typename... Args>
  static void trace(fmt::format_string<Args...> fmt, Args &&...args) {
    spdlog::trace(fmt, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static void debug(fmt::format_string<Args...> fmt, Args &&...args) {
    spdlog::debug(fmt, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static void info(fmt::format_string<Args...> fmt, Args &&...args) {
    spdlog::info(fmt, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static void warn(fmt::format_string<Args...> fmt, Args &&...args) {
    spdlog::warn(fmt, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static void error(fmt::format_string<Args...> fmt, Args &&...args) {
    spdlog::error(fmt, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static void critical(fmt::format_string<Args...> fmt, Args &&...args) {
    spdlog::critical(fmt, std::forward<Args>(args)...);
  }
};

} // namespace hl
