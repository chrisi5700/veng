/**
 * @file
 * @author chris
 * @brief Implementation of @ref veng::graph::ThreadPoolScheduler.
 *
 * @ingroup rendergraph
 */

#include <utility>
#include <veng/rendergraph/ThreadPoolScheduler.hpp>

namespace veng::graph
{
std::size_t ThreadPoolScheduler::default_thread_count() noexcept
{
	const unsigned hardware = std::thread::hardware_concurrency();
	return hardware == 0 ? 4 : hardware;
}

ThreadPoolScheduler::ThreadPoolScheduler(std::size_t thread_count)
{
	if (thread_count == 0)
	{
		thread_count = 1;
	}
	m_threads.reserve(thread_count);
	for (std::size_t i = 0; i < thread_count; ++i)
	{
		m_threads.emplace_back([this]() { worker(); });
	}
}

ThreadPoolScheduler::~ThreadPoolScheduler()
{
	{
		const std::lock_guard lock(m_mutex);
		m_stop = true;
	}
	m_cv.notify_all();
	for (std::thread& thread : m_threads)
	{
		if (thread.joinable())
		{
			thread.join();
		}
	}
}

void ThreadPoolScheduler::submit(std::function<void()> task)
{
	{
		const std::lock_guard lock(m_mutex);
		m_tasks.push(std::move(task));
	}
	m_cv.notify_one();
}

void ThreadPoolScheduler::worker()
{
	for (;;)
	{
		std::function<void()> task;
		{
			std::unique_lock lock(m_mutex);
			m_cv.wait(lock, [this]() { return m_stop || !m_tasks.empty(); });
			if (m_stop && m_tasks.empty())
			{
				return;
			}
			task = std::move(m_tasks.front());
			m_tasks.pop();
		}
		task();
	}
}
} // namespace veng::graph
