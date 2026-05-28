# Add library definitions here
# See README.md for CMake library patterns and examples


add_library(veng_logging INTERFACE)
target_link_libraries(veng_logging
        INTERFACE spdlog::spdlog
)
target_compile_definitions(veng_logging
        INTERFACE
        LOG_FILE="${CMAKE_SOURCE_DIR}/logs/logs.txt"
)
target_include_directories(veng_logging
        INTERFACE
        ${CMAKE_SOURCE_DIR}/include
)

target_add_library(veng_context)
target_sources(veng_context PRIVATE context/Context.cpp context/VmaImpl.cpp)
target_link_libraries(veng_context
        PUBLIC Vulkan::Vulkan glfw veng_logging
               GPUOpen::VulkanMemoryAllocator
               unofficial::VulkanMemoryAllocator-Hpp::VulkanMemoryAllocator-Hpp
        PRIVATE vk-bootstrap::vk-bootstrap
)
target_compile_definitions(veng_context PUBLIC
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 # Vulkan specific defines
        VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL=1
        VULKAN_HPP_NO_EXCEPTIONS
        VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
        # VulkanMemoryAllocator-Hpp imports vk::operator</>/<= etc., which vulkan.hpp
        # only emits when the C++20 spaceship operator is disabled.
        VULKAN_HPP_NO_SPACESHIP_OPERATOR
        GLM_FORCE_DEPTH_ZERO_TO_ONE # depth different for vulkan as opposed to opengl
)
target_compile_definitions(veng_context PRIVATE
        # VMA loads Vulkan entry points through the dynamic dispatcher we feed it,
        # matching the VULKAN_HPP_DISPATCH_LOADER_DYNAMIC setup above.
        VMA_STATIC_VULKAN_FUNCTIONS=0
        VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)


target_add_library(veng_shader)
target_sources(veng_shader PRIVATE shader/Shader.cpp)
target_link_libraries(veng_shader PUBLIC veng_context slang::slang)
target_compile_definitions(veng_shader PUBLIC
        SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders"
)

target_add_library(veng_pipelines)
target_sources(veng_pipelines PRIVATE pipelines/ComputePipeline.cpp pipelines/GraphicsPipeline.cpp)
target_link_libraries(veng_pipelines PUBLIC veng_shader glm::glm)

# L2.2 descriptor allocation — growable pool allocator over the layouts pipelines expose.
target_add_library(veng_descriptors)
target_sources(veng_descriptors PRIVATE descriptors/DescriptorAllocator.cpp)
target_link_libraries(veng_descriptors PUBLIC veng_context)

# L1 GPU resources — RAII Buffer/Image over VMA, plus the versioned N-buffered ResourcePool.
target_add_library(veng_resources)
target_sources(veng_resources PRIVATE resources/Buffer.cpp resources/Image.cpp resources/ResourcePool.cpp)
target_link_libraries(veng_resources PUBLIC veng_context)

# GPU bridge (L3 -> Vulkan seam): header-only GpuExecContext + GpuNode.
add_library(veng_gpu INTERFACE)
target_link_libraries(veng_gpu INTERFACE veng_render_graph veng_context veng_resources)
target_include_directories(veng_gpu INTERFACE ${CMAKE_SOURCE_DIR}/include)

# L2.4 device-side managers — command pools (per queue/slot/thread) + the windowed
# swapchain (over vk-bootstrap) + the per-frame FrameExecutor driving the GPU loop.
target_add_library(veng_managers)
target_sources(veng_managers PRIVATE managers/CommandManager.cpp managers/SwapchainManager.cpp managers/FrameExecutor.cpp)
target_link_libraries(veng_managers PUBLIC veng_context veng_gpu PRIVATE vk-bootstrap::vk-bootstrap)

# L4 nodes — concrete GPU graph nodes for the rendering slice.
target_add_library(veng_nodes)
target_sources(veng_nodes PRIVATE nodes/BlitNode.cpp nodes/PresentNode.cpp nodes/GraphicsNode.cpp nodes/MeshNode.cpp nodes/DynamicMeshNode.cpp nodes/UniformNode.cpp nodes/StorageBufferNode.cpp nodes/ScreenshotNode.cpp)
target_link_libraries(veng_nodes PUBLIC veng_gpu veng_pipelines veng_managers veng_descriptors)

# L3 Reactive Core. Deliberately Vulkan-agnostic (design.md §3): no link to the
# Vulkan construction layers, so it builds and unit-tests without a GPU.
target_add_library(veng_render_graph)
target_sources(veng_render_graph PRIVATE rendergraph/Graph.cpp rendergraph/ThreadPoolScheduler.cpp)
target_include_directories(veng_render_graph PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(veng_render_graph PUBLIC Threads::Threads)

