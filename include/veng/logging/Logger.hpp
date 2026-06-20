/**
 * @file
 * @author chris
 * @brief Engine logging facade: a singleton `spdlog::logger` that writes to both the
 *        console (coloured) and a persistent log file.
 *
 * @ref veng::Logger extends `spdlog::logger` with two sinks — a coloured `stdout` sink and a
 * basic file sink — configured at compile time: in debug builds both sinks capture
 * `trace` and above; in release builds the console sink captures `warn` and above while
 * the file sink still captures `trace`. The call site is embedded in the log pattern via
 * `std::source_location` so every message is tagged with its filename and line number.
 *
 * @ingroup logging
 */

#ifndef VENG_LOGGER_HPP
#define VENG_LOGGER_HPP

#include <cstdlib>
#include <filesystem>
#include <source_location>
#include <spdlog/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace veng
{
/**
 * @brief Singleton engine logger: coloured `stdout` sink + persistent file sink.
 *
 * Inherits the full `spdlog::logger` API. In debug builds both sinks capture at
 * `trace` level; in release builds the console sink captures at `warn` and above
 * while the file sink captures at `trace` and above. Each call to `instance()` updates
 * the format pattern with the caller's filename and line number via
 * `std::source_location`.
 *
 * @ingroup logging
 */
class Logger : public spdlog::logger
{
	std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> console_sink; ///< Coloured stdout sink.
	std::shared_ptr<spdlog::sinks::basic_file_sink_mt>	 file_sink;	   ///< Persistent file sink.

	/// The log-file path: a `VENG_LOG_FILE` env override, else the compile-time `LOG_FILE` default
	/// (the source tree for a build/add_subdirectory consumer; a relocated install defaults it to
	/// "veng.log" in the working directory). Lets an installed package log somewhere writable.
	static const char* log_file_path() noexcept
	{
		const char* const override_path = std::getenv("VENG_LOG_FILE");
		return (override_path != nullptr && override_path[0] != '\0') ? override_path : LOG_FILE;
	}

	Logger()
		: logger("veng")
		, console_sink(std::make_shared<spdlog::sinks::stdout_color_sink_mt>())
		, file_sink(std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file_path(), true))

	{
#ifndef NDEBUG
		console_sink->set_level(spdlog::level::trace);
		file_sink->set_level(spdlog::level::trace);
#else
		console_sink->set_level(spdlog::level::warn);
		file_sink->set_level(spdlog::level::trace);
#endif
		sinks().push_back(console_sink);
		sinks().push_back(file_sink);
	}

	 public:
	/**
	 * @brief Access the singleton logger, updating the format pattern with the call site.
	 *
	 * The format pattern embeds the caller's filename and line number so each log message
	 * shows its origin without any extra boilerplate at the call site.
	 *
	 * @param loc Automatically captured call-site location.
	 * @return Reference to the engine-wide `Logger` singleton.
	 */
	static Logger& instance(std::source_location loc = std::source_location::current())
	{
		static Logger		  logger;
		std::filesystem::path path	   = loc.file_name();
		auto				  location = fmt::format("[{}:{}]", std::string{path.filename()}, loc.line());
		auto				  pattern  = fmt::format("[veng]{:<30}[%^%5l%$] %v", location);
		logger.set_pattern(pattern);
		return logger;
	}
};
} // namespace veng

#endif // VENG_LOGGER_HPP
