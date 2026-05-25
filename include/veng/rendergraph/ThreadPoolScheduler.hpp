//
// Created by chris on 5/25/26.
//
// L5 job system — a fixed-size worker pool implementing the L3 Scheduler interface
// (design.md §L5, §7: thread-per-node is non-viable at 60–144 Hz). Vulkan-agnostic.
//

#ifndef VENG_THREADPOOLSCHEDULER_HPP
#define VENG_THREADPOOLSCHEDULER_HPP

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <veng/rendergraph/nodes/Node.hpp>

namespace veng::graph
{
/// A pool of worker threads that drains a shared task queue. Paired with the
/// Graph's height-batched dispatch, every task submitted within a height band is
/// independent of the others, so no task ever blocks waiting on another task — which
/// means this fixed-size pool cannot deadlock no matter how deep the graph is.
///
/// A single shared queue (rather than per-worker work-stealing deques) is plenty for
/// v1 correctness; work-stealing is a throughput optimization noted for later.
class ThreadPoolScheduler final : public Scheduler
{
	 public:
	explicit ThreadPoolScheduler(std::size_t thread_count = default_thread_count());
	ThreadPoolScheduler(const ThreadPoolScheduler&)			   = delete;
	ThreadPoolScheduler& operator=(const ThreadPoolScheduler&) = delete;
	ThreadPoolScheduler(ThreadPoolScheduler&&)				   = delete;
	ThreadPoolScheduler& operator=(ThreadPoolScheduler&&)	   = delete;
	~ThreadPoolScheduler() override;

	void submit(std::function<void()> task) override;

	[[nodiscard]] std::size_t thread_count() const noexcept { return m_threads.size(); }

	/// Worker count when none is given: the hardware concurrency, or 4 if unknown.
	[[nodiscard]] static std::size_t default_thread_count() noexcept;

	 private:
	void worker();

	std::vector<std::thread>		  m_threads;
	std::queue<std::function<void()>> m_tasks;
	std::mutex						  m_mutex;
	std::condition_variable			  m_cv;
	bool							  m_stop = false;
};
} // namespace veng::graph

#endif // VENG_THREADPOOLSCHEDULER_HPP
