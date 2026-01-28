//
// Created by chris on 1/24/26.
//
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/nodes/Node.hpp>


const std::atomic<State>& Data::request()
{
	auto parent = get_parent();
	if (not parent)
		throw std::logic_error{"No parent node"};
	return parent->process();
}

bool Data::valid()
{
	auto parent = get_parent();
	if (not parent)
		return true;
	return parent->valid();
}



