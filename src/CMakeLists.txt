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
        # With NO_EXCEPTIONS the engine hand-checks every `.result`; without this, vulkan-hpp's
        # convenience wrappers (createImageView/createDescriptorPool/createPipeline/...) still
        # assert-abort on failure in debug, so those std::expected error branches were dead code
        # under the gate (asserts on). Emptying the assert lets resultCheck fall through and the
        # engine's own error handling run — a recoverable Vulkan error returns instead of crashing.
        VULKAN_HPP_ASSERT_ON_RESULT=
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
target_sources(veng_nodes PRIVATE nodes/BlitNode.cpp nodes/PresentNode.cpp nodes/GraphicsNode.cpp nodes/MeshNode.cpp nodes/DynamicMeshNode.cpp nodes/UniformNode.cpp nodes/StorageBufferNode.cpp nodes/ScreenshotNode.cpp nodes/MeshSelectorNode.cpp nodes/CoverageLodNode.cpp)
target_link_libraries(veng_nodes PUBLIC veng_gpu veng_pipelines veng_managers veng_descriptors)

# Clustered-shading culling math — pure glm, no Vulkan (unit-tests on the CPU without a device, the
# same discipline as veng_render_graph). Froxel AABB build + light-to-cluster assignment; the graph
# wires these behind FroxelGridCpu / LightCullCpu so a GPU implementation is a drop-in swap.
target_add_library(veng_culling)
target_sources(veng_culling PRIVATE culling/Clusters.cpp)
target_link_libraries(veng_culling PUBLIC glm::glm)
target_include_directories(veng_culling PUBLIC ${CMAKE_SOURCE_DIR}/include)

# L5 passes — reusable, higher-level effects composed from L4 nodes (the engine's pass
# library, shaders under shaders/passes/). The screen-space outline (silhouette -> separable
# gaussian blur -> ring extract) is the first entry.
target_add_library(veng_passes)
target_sources(veng_passes PRIVATE passes/OutlinePass.cpp passes/PickingPass.cpp passes/PhongPass.cpp passes/PbrPass.cpp passes/ClusteredLights.cpp)
target_link_libraries(veng_passes PUBLIC veng_nodes veng_culling glm::glm)

# L2 assets — file/memory-backed GPU resources for displaying real content. The texture loader
# (stb_image decode -> staging -> device-local -> generated mip chain) lands first; the glTF
# scene loader (fastgltf -> MeshNodes + material SSBO + textures) is added to this target in the
# glTF phase. Depends on the resource + manager layers and on veng_nodes (the glTF loader emits
# L4 nodes).
find_package(Stb REQUIRED)
find_package(fastgltf CONFIG REQUIRED)
find_package(meshoptimizer CONFIG REQUIRED)
target_add_library(veng_assets)
target_sources(veng_assets PRIVATE assets/Texture.cpp assets/StbImpl.cpp assets/GltfLoader.cpp assets/StlLoader.cpp assets/MeshProcessing.cpp)
# meshoptimizer (decimation) is an implementation detail of the loaders — PRIVATE so it never leaks
# into a public header, exactly like fastgltf.
target_link_libraries(veng_assets PUBLIC veng_resources veng_managers veng_nodes glm::glm PRIVATE fastgltf::fastgltf meshoptimizer::meshoptimizer)
target_include_directories(veng_assets PRIVATE ${Stb_INCLUDE_DIR})
# stb_image is third-party and not -Wall/-Werror clean; silence warnings for its one impl TU.
set_source_files_properties(assets/StbImpl.cpp PROPERTIES COMPILE_OPTIONS "-w")

# L3 Reactive Core. Deliberately Vulkan-agnostic: no link to the
# Vulkan construction layers, so it builds and unit-tests without a GPU.
target_add_library(veng_render_graph)
target_sources(veng_render_graph PRIVATE rendergraph/Graph.cpp rendergraph/ThreadPoolScheduler.cpp)
target_include_directories(veng_render_graph PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(veng_render_graph PUBLIC Threads::Threads)

