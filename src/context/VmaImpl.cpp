//
// Single translation unit that compiles the VulkanMemoryAllocator implementation.
//
// VMA is header-only: exactly one source file in the whole project must define
// VMA_IMPLEMENTATION before including the header. The C++ bindings
// (vk_mem_alloc.hpp, namespace vma) are inline and live wherever they are
// included, so only the underlying C implementation needs to be built here.
//
// The function-loading macros (VMA_STATIC_VULKAN_FUNCTIONS=0,
// VMA_DYNAMIC_VULKAN_FUNCTIONS=1) are set on the veng_context target so VMA
// fetches its entry points through the same dynamic dispatcher Context feeds it.

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
