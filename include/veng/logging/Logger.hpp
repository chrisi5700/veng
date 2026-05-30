//
// Created by chris on 1/22/26.
//

#ifndef VENG_LOGGER_HPP
#define VENG_LOGGER_HPP

#include <filesystem>
#include <source_location>
#include <spdlog/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace veng
{
class Logger : public spdlog::logger
{
	std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> console_sink;
	std::shared_ptr<spdlog::sinks::basic_file_sink_mt>	 file_sink;
	Logger()
		: logger("veng")
		, console_sink(std::make_shared<spdlog::sinks::stdout_color_sink_mt>())
		, file_sink(std::make_shared<spdlog::sinks::basic_file_sink_mt>(LOG_FILE, true))

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
