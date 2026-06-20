/**
 * @file
 * @author chris
 * @brief Umbrella header: include `<veng/veng.hpp>` to pull in veng's public API.
 *
 * Convenience include for consumers who would otherwise reach into veng's subdirectory layout by
 * hand. It aggregates the public surface — context, the reactive render graph, resources, the
 * device-side managers, the GPU seam (including the host-blit interop helper), the standard nodes,
 * the RHI vocabulary, pipelines, the pass library, and the asset loaders. Pair it with the
 * `veng::veng` CMake target (`target_link_libraries(my_app PRIVATE veng::veng)`), which carries all
 * the usage requirements. Include the narrower per-subsystem headers directly if you prefer to keep
 * a translation unit's include cost down.
 *
 * @defgroup gpu_handles GPU handles
 */

#ifndef VENG_VENG_HPP
#define VENG_VENG_HPP

// Engine context (device/instance creation, or host adoption) + its typed error variant.
#include <veng/context/Context.hpp>
#include <veng/context/ContextErrors.hpp>

// L3 reactive render graph: the graph, data nodes, the node/scheduler interface, helpers.
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/Graph.hpp>
#include <veng/rendergraph/nodes/Node.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <veng/rendergraph/Resolve.hpp>
#include <veng/rendergraph/ThreadPoolScheduler.hpp>

// L1 GPU resources + the N-buffered transient pool.
#include <veng/resources/Buffer.hpp>
#include <veng/resources/Image.hpp>
#include <veng/resources/RenderTargetSet.hpp>
#include <veng/resources/ResourcePool.hpp>

// L2 device-side managers: command pools, the windowed swapchain, the frame/headless executors.
#include <veng/managers/CommandManager.hpp>
#include <veng/managers/FrameExecutor.hpp>
#include <veng/managers/HeadlessExecutor.hpp>
#include <veng/managers/QueueKind.hpp>
#include <veng/managers/SwapchainManager.hpp>

// GPU seam: the per-frame exec context, edge value types, and the host-interop blit helper.
#include <veng/gpu/BufferRef.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/gpu/HostBlit.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/MeshRef.hpp>
#include <veng/gpu/Sink.hpp>
#include <veng/gpu/SubmitContext.hpp>
#include <veng/gpu/UniformRef.hpp>
#include <veng/gpu/VersionedOutput.hpp>
#include <veng/gpu/Vertex.hpp>

// L4 standard nodes.
#include <veng/nodes/BlitNode.hpp>
#include <veng/nodes/CoverageLodNode.hpp>
#include <veng/nodes/DynamicMeshNode.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/nodes/MeshNode.hpp>
#include <veng/nodes/MeshSelectorNode.hpp>
#include <veng/nodes/PresentNode.hpp>
#include <veng/nodes/ScreenshotNode.hpp>
#include <veng/nodes/StorageBufferNode.hpp>
#include <veng/nodes/UniformNode.hpp>

// RHI vocabulary (handles, enums, conversions, the device registry, the command encoder).
#include <veng/rhi/BindGroup.hpp>
#include <veng/rhi/CommandEncoder.hpp>
#include <veng/rhi/Convert.hpp>
#include <veng/rhi/Device.hpp>
#include <veng/rhi/Enums.hpp>
#include <veng/rhi/Error.hpp>
#include <veng/rhi/Handles.hpp>
#include <veng/rhi/SamplerDesc.hpp>
#include <veng/rhi/Swapchain.hpp>

// Pipelines, descriptors, shaders, logging, culling math.
#include <veng/culling/Clusters.hpp>
#include <veng/descriptors/DescriptorAllocator.hpp>
#include <veng/logging/Logger.hpp>
#include <veng/pipelines/ComputePipeline.hpp>
#include <veng/pipelines/GraphicsPipeline.hpp>
#include <veng/pipelines/PipelineError.hpp>
#include <veng/shader/Shader.hpp>

// L5 pass library (higher-level effects composed from L4 nodes).
#include <veng/passes/ClusteredLights.hpp>
#include <veng/passes/OutlinePass.hpp>
#include <veng/passes/PbrPass.hpp>
#include <veng/passes/PhongPass.hpp>
#include <veng/passes/PickingPass.hpp>

// L2 asset loaders (glTF / OBJ / PLY / STL / 3MF meshes + textures). Heavier to compile; include the
// individual headers instead if a translation unit does not need the loaders.
#include <veng/assets/GltfLoader.hpp>
#include <veng/assets/Lod.hpp>
#include <veng/assets/MeshData.hpp>
#include <veng/assets/ObjLoader.hpp>
#include <veng/assets/PlyLoader.hpp>
#include <veng/assets/StlLoader.hpp>
#include <veng/assets/Texture.hpp>
#include <veng/assets/ThreemfLoader.hpp>

#endif // VENG_VENG_HPP
