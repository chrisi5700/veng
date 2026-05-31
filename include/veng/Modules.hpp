/**
 * @file
 * @author chris
 * @brief Doxygen module (group) hierarchy for the veng engine.
 *
 * This header contains no code — only the `@defgroup` definitions that organise the
 * generated documentation into a cohesive module tree. Every documented entity declares
 * `@ingroup <id>` against one of the groups defined here so the API reference is browsable
 * by subsystem rather than as a flat symbol list.
 */

#ifndef VENG_MODULES_HPP
#define VENG_MODULES_HPP

/**
 * @defgroup veng veng
 * @brief A reactive, render-graph-driven Vulkan rendering engine.
 *
 * The engine is layered: low-level Vulkan resource wrappers (@ref resources, @ref gpu_handles)
 * are composed by a reactive render graph (@ref rendergraph) whose @ref graph_nodes and
 * higher-level @ref render_passes describe a frame declaratively. The @ref context and
 * @ref managers layers own the device, swapchain, and per-frame submission.
 */

/**
 * @defgroup util Utilities
 * @ingroup veng
 * @brief Header-only helpers shared across the engine (error merging, visitors).
 */

/**
 * @defgroup context Vulkan Context
 * @ingroup veng
 * @brief Instance/device/allocator creation and the engine-wide @ref veng::Context handle.
 */

/**
 * @defgroup resources Vulkan Resource Wrappers
 * @ingroup veng
 * @brief RAII wrappers over images and buffers, plus the versioned transient @ref veng::ResourcePool.
 */

/**
 * @defgroup gpu_handles GPU Resource Handles
 * @ingroup veng
 * @brief Lightweight, copyable handles (`*Ref`) and the GPU execution/submit contexts that
 *        pass GPU resources between render-graph nodes.
 */

/**
 * @defgroup rendergraph Reactive Render Graph
 * @ingroup veng
 * @brief The engine-neutral core: data edges, nodes, the graph, and its scheduler.
 */

/**
 * @defgroup graph_nodes Render Graph Nodes
 * @ingroup veng
 * @brief Concrete GPU nodes (draws, uploads, present, blit, readback) wired into a graph.
 */

/**
 * @defgroup render_passes Reusable Render Passes
 * @ingroup veng
 * @brief Higher-level, configurable passes (PBR, Phong, outline, picking, clustered lights)
 *        assembled from @ref graph_nodes.
 */

/**
 * @defgroup pipelines Pipelines
 * @ingroup veng
 * @brief Graphics and compute pipeline objects built from reflected shaders.
 */

/**
 * @defgroup descriptors Descriptors
 * @ingroup veng
 * @brief Descriptor-set allocation and pooling.
 */

/**
 * @defgroup shaders Shader Loading & Reflection
 * @ingroup veng
 * @brief Shader-module loading and SPIR-V reflection.
 */

/**
 * @defgroup managers Frame & Swapchain Management
 * @ingroup veng
 * @brief Swapchain, command-pool, and per-frame submission management.
 */

/**
 * @defgroup assets Asset Loading
 * @ingroup veng
 * @brief glTF and image/texture loading.
 */

/**
 * @defgroup culling Light Culling
 * @ingroup veng
 * @brief Froxel/cluster construction for clustered-forward lighting.
 */

/**
 * @defgroup logging Logging
 * @ingroup veng
 * @brief Engine logging facade.
 */

#endif // VENG_MODULES_HPP
