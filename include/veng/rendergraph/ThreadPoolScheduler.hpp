/**
 * @file
 * @author chris
 * @brief Fixed-size worker-thread pool implementing the @ref veng::graph::Scheduler interface.
 *
 * A thread-per-node approach is non-viable at 60–144 Hz; this pool amortises
 * thread creation over the engine lifetime. Vulkan-agnostic.
 *
 * @ingroup rendergraph
 */

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
/**
 * @brief A pool of worker threads that drains a shared task queue.
 *
 * Paired with @ref veng::graph::Graph's height-batched dispatch, every task submitted within a height
 * band is independent of every other task in that band, so no task ever blocks waiting on
 * another task — which means this fixed-size pool cannot deadlock no matter how deep the
 * graph is.
 *
 * A single shared queue (rather than per-worker work-stealing deques) is sufficient for v1
 * correctness; work-stealing is a throughput optimisation noted for a later revision.
 *
 * @ingroup rendergraph
 * @see Scheduler
 * @see InlineScheduler
 * @see Graph::execute
 */
class ThreadPoolScheduler final : public Scheduler
{
	 public:
	/**
	 * @brief Construct and start `thread_count` worker threads.
	 * @param thread_count Number of worker threads; defaults to @ref default_thread_count().
	 *                     Clamped to at least 1.
	 */
	explicit ThreadPoolScheduler(std::size_t thread_count = default_thread_count());
	ThreadPoolScheduler(const ThreadPoolScheduler&)			   = delete;
	ThreadPoolScheduler& operator=(const ThreadPoolScheduler&) = delete;
	ThreadPoolScheduler(ThreadPoolScheduler&&)				   = delete;
	ThreadPoolScheduler& operator=(ThreadPoolScheduler&&)	   = delete;

	/// @brief Signal workers to stop and join all threads.
	~ThreadPoolScheduler() override;

	/**
	 * @brief Enqueue a task and wake one worker thread.
	 * @param task The callable to run on a worker thread.
	 */
	void submit(std::function<void()> task) override;

	/// @brief Returns the number of worker threads in this pool.
	[[nodiscard]] std::size_t thread_count() const noexcept { return m_threads.size(); }

	/**
	 * @brief Default worker count: `std::thread::hardware_concurrency()`, or 4 if unknown.
	 * @return The recommended thread count for the current hardware.
	 */
	[[nodiscard]] static std::size_t default_thread_count() noexcept;

	 private:
	/// @brief Worker thread body: dequeues and runs tasks until @ref m_stop is set.
	void worker();

	std::vector<std::thread>		  m_threads;	  ///< Live worker threads.
	std::queue<std::function<void()>> m_tasks;		  ///< Shared task queue (guarded by @ref m_mutex).
	std::mutex						  m_mutex;		  ///< Protects @ref m_tasks and @ref m_stop.
	std::condition_variable			  m_cv;			  ///< Wakes workers when a task arrives or stop is set.
	bool							  m_stop = false; ///< True once the destructor has been entered.
};
} // namespace veng::graph

#endif // VENG_THREADPOOLSCHEDULER_HPP
