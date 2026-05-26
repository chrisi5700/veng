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
// Like ImageRef/MeshRef, deliberately NOT equality-comparable (it has a std::string member
// but no operator==): `ValueData<T>` then treats every re-produce as "changed" (Data.hpp).
// A uniform's buffer handle is stable across re-uploads (the contents change in place), so
// a comparable ref would be mistaken for "unchanged" and the dependent draw cached away —
// exactly the staleness we must avoid.
//

#ifndef VENG_UNIFORMREF_HPP
#define VENG_UNIFORMREF_HPP

#include <string>
#include <vulkan/vulkan.hpp>

namespace veng::gpu
{
struct UniformRef
{
	vk::Buffer	   buffer{};
	vk::DeviceSize size = 0;
	std::string	   name; // the reflected descriptor binding this fills (DescriptorInfo.name)

	// No operator== on purpose — see the file header.
};
} // namespace veng::gpu

#endif // VENG_UNIFORMREF_HPP
