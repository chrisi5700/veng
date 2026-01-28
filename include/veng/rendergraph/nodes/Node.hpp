//
// Created by chris on 1/24/26.
//

#ifndef VENG_NODE_HPP
#define VENG_NODE_HPP
#include <vector>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <functional>
#include <atomic>
#include <vulkan/vulkan.hpp>
class Data;


class CPUNode
{
	std::atomic<State> m_state = State::Invalid;

protected:
	virtual void invalidate_output() = 0;
	virtual void schedule(std::function<void()> f) = 0;
	virtual std::vector<const std::atomic<State>*> get_pending_input() = 0;
	virtual void								   execute()		   = 0;

public:
	[[nodiscard]] bool valid() const { return m_state == State::Valid; }
	void invalidate();
	[[nodiscard]] std::atomic<State>& process();

	virtual ~CPUNode() = default;
};

struct GPUSync
{
	std::uint64_t expected_val;
	vk::Semaphore semaphore;
};

class GPUNode
{
	vk::Semaphore m_semaphore; // Timeline semaphore
	std::uint64_t m_next_val = 1;
	bool m_submitted = false;
protected:
	virtual void invalidate_output() = 0;
	virtual std::vector<GPUSync> get_pending_input() = 0;
	virtual void								   execute()		   = 0;
public:

};
#endif // VENG_NODE_HPP
