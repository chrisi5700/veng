//
// Created by chris on 5/26/26.
//
// The value that flows on a *uniform* edge of the reactive graph (design.md §L4) — the
// descriptor-set counterpart to MeshRef. It is a non-owning *reference* to a GPU uniform
// buffer plus the reflected binding *name* it fills. A UniformNode owns the backing
// `Buffer`, uploads the value, and produces a `UniformRef`; a `GraphicsNode::add_uniform`
// reads it, matches `name` to the shader's reflected descriptor (DescriptorInfo.name), and
// writes the buffer into a descriptor set. This is the user's `UniformNode{value, "name"}
// -> GraphicsNode.add_uniform(...)`.
//
// Equality is value-based, including a `version` the producing `UniformNode` bumps on every
// `produce` — critical because the buffer handle is *stable* across re-uploads (contents
// change in place), so without a version a structural comparison would call two distinct
// upload values "equal" and silently cache away the consuming draw.
//

#ifndef VENG_UNIFORMREF_HPP
#define VENG_UNIFORMREF_HPP

#include <cstdint>
#include <string>
#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
struct UniformRef
{
	vk::Buffer	   buffer{};
	vk::DeviceSize size = 0;
	std::string	   name; // the reflected descriptor binding this fills (DescriptorInfo.name)

	// Producer-bumped version (see the file header).
	std::uint64_t version = 0;

	friend bool operator==(const UniformRef&, const UniformRef&) noexcept = default;
};
} // namespace veng::gpu

#endif // VENG_UNIFORMREF_HPP
