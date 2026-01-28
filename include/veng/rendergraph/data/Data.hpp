//
// Created by chris on 1/24/26.
//

#ifndef VENG_DATA_HPP
#define VENG_DATA_HPP
#include <memory>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <atomic>

class CPUNode;


class Data
{
public:
	virtual CPUNode*			  get_parent() = 0;
	virtual void			  invalidate() = 0;
	[[nodiscard]] const std::atomic<State>& request();
	bool					  valid();

	virtual ~Data() = default;
};

#endif // VENG_DATA_HPP
