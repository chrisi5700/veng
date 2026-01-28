//
// Created by chris on 1/24/26.
//
#include <veng/rendergraph/nodes/Node.hpp>
#include <veng/rendergraph/data/Data.hpp>
void CPUNode::invalidate()
{
	m_state.store(State::Invalid, std::memory_order_relaxed);
	invalidate_output();
}
std::atomic<State>& CPUNode::process()
{
	auto expected = State::Invalid;
	if (!m_state.compare_exchange_strong(expected, State::Processing, std::memory_order_acquire,
									   std::memory_order_relaxed))
	{
		return m_state;
	}

	for (auto* state : get_pending_input())
	{
		state->wait(State::Processing, std::memory_order_acquire);
	}

	schedule(
		[&]()
		{
			execute();
			m_state.store(State::Valid, std::memory_order_release);
			m_state.notify_all();
		});
	return m_state;
}