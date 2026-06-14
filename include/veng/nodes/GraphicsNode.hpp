/**
 * @file
 * @author chris
 * @brief L4 generic raster node: self-contained pipeline, multi-draw, reactive uniforms and
 *        sampled-image inputs, pool-backed N-buffered render target.
 *
 * The node is self-contained: you give it the names of a vertex and fragment shader, the
 * attachment formats, a screen-size input that sizes its owned color (and optional depth)
 * target, and zero or more push-constant value edges. It builds its own pipeline internally
 * on the first record when it has a device — graph construction never touches the L2 pipeline
 * layer — then draws and publishes the rendered target as an `ImageData`. A cube scene is just
 * this node wired to the cube shaders plus an `mvp` push-constant edge fed by a pure transform
 * of the angle: no cube-specific node. A resize reallocates the target (an ordinary
 * invalidation); a changed push-constant re-renders, an unchanged one is cached.
 *
 * Geometry is either buffer-backed — @ref veng::nodes::GraphicsNode::set_mesh "set_mesh(handle)" binds a
 * `gpu::MeshRef` produced by a @ref veng::nodes::MeshNode and the node draws it (indexed when the ref has
 * an index buffer) — or, with no mesh bound, a `vertex_count` draw of vertices fabricated in
 * the shader from `SV_VertexID` (a fullscreen post-process triangle is just `vertex_count = 3`
 * plus a fullscreen vertex shader). One node can host several draws via
 * @ref veng::nodes::GraphicsNode::add_draw "add_draw(mesh)" — they share its pipeline, target, uniforms and
 * sampled images, so N meshes land in one pass (a scene), each with its own push constants
 * (e.g. a per-object model matrix). @ref veng::nodes::GraphicsNode::set_mesh "set_mesh" and
 * @ref veng::nodes::GraphicsNode::push_constant "push_constant" are the single-draw shorthand for the first
 * such draw.
 *
 * @ref veng::nodes::GraphicsNode::add_uniform "add_uniform(handle)" binds a `gpu::UniformRef` (from a
 * @ref veng::nodes::UniformNode) into the node's own descriptor set, matched to the shader's reflected
 * binding by name. @ref veng::nodes::GraphicsNode::add_sampled_image "add_sampled_image(handle, name)"
 * binds a `gpu::ImageRef` as a sampled texture (matched the same way, with a node-owned
 * sampler) — this is what lets one pass read another's output, the basis of multi-pass effects.
 * @ref veng::nodes::GraphicsNode::set_sampled_image "set_sampled_image(name, handle)" rebinds such an input
 * at runtime: a rebind that drops an upstream branch makes that branch undemanded, so it stops
 * executing.
 *
 * @ingroup graph_nodes
 * @see MeshNode
 * @see UniformNode
 */

#ifndef VENG_GRAPHICSNODE_HPP
#define VENG_GRAPHICSNODE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <veng/descriptors/DescriptorAllocator.hpp>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/gpu/VersionedOutput.hpp>
#include <veng/pipelines/GraphicsPipeline.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <veng/resources/Image.hpp>
#include <veng/resources/RenderTargetSet.hpp>
#include <veng/resources/SamplerConfig.hpp>
#include <veng/rhi/Enums.hpp>
#include <veng/shader/Shader.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::nodes
{
/**
 * @brief L4 generic raster node: owns its pipeline, render target, and descriptor set; supports
 *        multi-draw, reactive uniforms, push constants, and sampled-image inputs.
 * @ingroup graph_nodes
 * @see MeshNode
 * @see UniformNode
 * @see DrawConfig
 */
class GraphicsNode final : public gpu::GpuNode
{
	 public:
	/**
	 * @brief Construct a graphics node.
	 *
	 * `vertex_shader` and `fragment_shader` are shader names resolved from `SHADER_DIR`; the
	 * node loads them and builds its pipeline for (`color_format`[, `depth_format`]) on first
	 * record. Pass `eUndefined` for `depth_format` to render without a depth attachment.
	 * `screen_size` is a `ValueData<vk::Extent2D>` source that sizes the owned render target.
	 * `output` is the scene `ImageData` edge the rendered color target is published on.
	 * `vertex_count` is used for `SV_VertexID` draws when no mesh is bound.
	 *
	 * @param vertex_shader   Shader name (resolved from `SHADER_DIR`) for the vertex stage.
	 * @param fragment_shader Shader name for the fragment stage.
	 * @param color_format    Format of the owned color render target.
	 * @param depth_format    Format of the owned depth target; pass `eUndefined` for no depth.
	 * @param vertex_count    Fallback vertex count for `SV_VertexID` draws (no mesh bound).
	 * @param screen_size     `ValueData<vk::Extent2D>` edge that sizes the render target.
	 * @param output          `ImageData` edge this node publishes its color target on.
	 */
	GraphicsNode(std::string vertex_shader, std::string fragment_shader, rhi::Format color_format,
				 rhi::Format depth_format, std::uint32_t vertex_count, graph::DataHandle screen_size,
				 graph::DataHandle output) noexcept;

	/// Holds reflected descriptor state + a pipeline; non-copyable and non-movable. Its sampler is
	/// owned by @ref veng::rhi::Device (freed with the device), so no destructor work is needed.
	GraphicsNode(const GraphicsNode&)			 = delete;
	GraphicsNode& operator=(const GraphicsNode&) = delete;
	GraphicsNode(GraphicsNode&&)				 = delete;
	GraphicsNode& operator=(GraphicsNode&&)		 = delete;
	~GraphicsNode()								 = default;

	/**
	 * @brief Bind a push-constant value edge to the default (first) draw.
	 *
	 * `T` is the value type on `handle`; its bytes are pushed at `offset` for `stage` each
	 * time the node records. The edge is a reactive input — changing it re-renders the node.
	 * Call before adding the node to the graph (it extends the input set).
	 *
	 * @tparam T           The push-constant value type (deduced from the edge).
	 * @param handle       `ValueData<T>` edge whose value is pushed each record.
	 * @param stage        Shader stage(s) the constant is visible to (default: vertex).
	 * @param offset       Byte offset within the push-constant range (default: 0).
	 * @return `*this` for chaining.
	 */
	template <class T>
	GraphicsNode& push_constant(graph::DataHandle handle, rhi::ShaderStage stage = rhi::ShaderStage::VERTEX,
								std::uint32_t offset = 0)
	{
		default_draw().push_constants.push_back(make_push<T>(handle, stage, offset));
		m_inputs.push_back(handle); // a push-constant edge is a dirtiness input
		mark_dirty();				// extending m_inputs at runtime must re-plan us
		return *this;
	}

	/**
	 * @brief Bind a mesh edge to the default (first) draw.
	 *
	 * The node draws the buffer-backed geometry on `handle` (a `ValueData<gpu::MeshRef>`,
	 * produced by a @ref veng::nodes::MeshNode) instead of fabricating vertices from `SV_VertexID`. The
	 * draw is indexed if the ref carries an index buffer, else a non-indexed draw of the
	 * ref's vertex count — either way overriding the constructor's `vertex_count`. The vertex
	 * layout must match the vertex shader's reflected input. A reactive input: re-uploading
	 * the mesh re-renders. Call before adding the node to the graph (it extends the input set).
	 *
	 * @param handle `ValueData<gpu::MeshRef>` edge produced by a @ref veng::nodes::MeshNode.
	 * @return `*this` for chaining.
	 */
	GraphicsNode& set_mesh(graph::DataHandle handle)
	{
		default_draw().mesh = handle;
		m_inputs.push_back(handle);
		mark_dirty();
		return *this;
	}

	/**
	 * @brief Per-draw configuration returned by @ref add_draw.
	 *
	 * Lets the caller attach push-constant edges (e.g. a per-object model matrix) to one
	 * specific draw. Pass-level state — uniforms, sampled images, the render target, clear
	 * color — lives on the @ref veng::nodes::GraphicsNode and is shared by every draw.
	 *
	 * @ingroup graph_nodes
	 * @see GraphicsNode::add_draw
	 */
	class DrawConfig
	{
		 public:
		/**
		 * @brief Bind a push-constant edge to this draw only.
		 *
		 * See @ref veng::nodes::GraphicsNode::push_constant for the full contract. The edge is a reactive
		 * input — changing it re-renders the pass.
		 *
		 * @tparam T      The push-constant value type.
		 * @param handle  `ValueData<T>` edge whose value is pushed each record.
		 * @param stage   Shader stage(s) the constant is visible to (default: vertex).
		 * @param offset  Byte offset within the push-constant range (default: 0).
		 * @return `*this` for chaining.
		 */
		template <class T>
		DrawConfig& push_constant(graph::DataHandle handle, rhi::ShaderStage stage = rhi::ShaderStage::VERTEX,
								  std::uint32_t offset = 0)
		{
			m_node->m_draws[m_index].push_constants.push_back(GraphicsNode::make_push<T>(handle, stage, offset));
			m_node->m_inputs.push_back(handle);
			m_node->mark_dirty();
			return *this;
		}

		/**
		 * @brief Set this draw's `instanceCount` to a fixed value.
		 *
		 * The shader reads `SV_InstanceID` to identify which instance it is rendering. For a
		 * non-instanced draw leave this unset (default `instanceCount` is 1).
		 *
		 * @param count The fixed instance count.
		 * @return `*this` for chaining.
		 */
		DrawConfig& set_instances(std::uint32_t count) noexcept
		{
			m_node->m_draws[m_index].instance_count = count;
			m_node->m_draws[m_index].instance_count_source.reset();
			m_node->mark_dirty();
			return *this;
		}

		/**
		 * @brief Drive this draw's `instanceCount` reactively from a `ValueData<gpu::BufferRef>` edge.
		 *
		 * Typically this is the same storage buffer that supplies the per-instance data. Each
		 * record reads `BufferRef::count` to decide how many instances to emit; a resize of the
		 * source array is therefore a single producer write that drives both the upload and the
		 * draw size. Adds the edge as a reactive input so a count change re-runs the draw.
		 *
		 * @param handle `ValueData<gpu::BufferRef>` edge whose `count` field drives `instanceCount`.
		 * @return `*this` for chaining.
		 */
		DrawConfig& set_instances_from(graph::DataHandle handle)
		{
			m_node->m_draws[m_index].instance_count_source = handle;
			m_node->m_inputs.push_back(handle);
			m_node->mark_dirty();
			return *this;
		}

		 private:
		friend class GraphicsNode;
		DrawConfig(GraphicsNode* node, std::size_t index) noexcept
			: m_node(node)
			, m_index(index)
		{
		}
		GraphicsNode* m_node;
		std::size_t	  m_index;
	};

	/**
	 * @brief Append a draw to this pass.
	 *
	 * With a valid `mesh` (a `ValueData<gpu::MeshRef>`, from a @ref veng::nodes::MeshNode) the draw uses
	 * that buffer-backed geometry (indexed when the ref carries indices); otherwise it draws
	 * `vertex_count` `SV_VertexID` vertices. Every draw shares the node's pipeline, render
	 * target, uniforms and sampled images — so this is how N meshes render into one pass, each
	 * with its own transform. Call before adding the node to the graph (it extends the input set).
	 *
	 * @param mesh         Optional `ValueData<gpu::MeshRef>` edge; empty handle means `SV_VertexID`.
	 * @param vertex_count `SV_VertexID` vertex count used when `mesh` is empty.
	 * @return @ref DrawConfig to attach per-draw push constants.
	 */
	DrawConfig add_draw(graph::DataHandle mesh = {}, std::uint32_t vertex_count = 0)
	{
		Draw draw;
		draw.vertex_count = vertex_count;
		if (mesh.valid())
		{
			draw.mesh = mesh;
			m_inputs.push_back(mesh);
		}
		m_draws.push_back(std::move(draw));
		mark_dirty();
		return DrawConfig{this, m_draws.size() - 1};
	}

	/**
	 * @brief Bind a uniform-buffer edge into this node's descriptor set.
	 *
	 * `handle` is a `ValueData<gpu::UniformRef>` produced by a @ref veng::nodes::UniformNode. It is written
	 * into this node's descriptor set at the binding whose reflected name matches the ref's
	 * `name`. A reactive input — changing the uniform value re-renders the node. Call before
	 * adding the node to the graph (it extends the input set).
	 *
	 * @param handle `ValueData<gpu::UniformRef>` edge from a @ref veng::nodes::UniformNode.
	 * @return `*this` for chaining.
	 */
	GraphicsNode& add_uniform(graph::DataHandle handle)
	{
		m_uniforms.push_back(handle);
		m_inputs.push_back(handle);
		mark_dirty();
		return *this;
	}

	/**
	 * @brief Bind a storage-buffer (SSBO) edge into this node's descriptor set.
	 *
	 * `handle` is a `ValueData<gpu::BufferRef>` produced by a `StorageBufferNode`. It is
	 * written into this node's descriptor set at the binding whose reflected name matches the
	 * ref's `name` — the shader sees it as a `StructuredBuffer<T>`. A reactive input:
	 * re-uploading the array re-renders the node. The published ref's `count` is what
	 * @ref DrawConfig::set_instances_from reads to drive `instanceCount`, so the same edge can
	 * also configure the draw size. Call before adding the node to the graph (it extends the
	 * input set).
	 *
	 * @param handle `ValueData<gpu::BufferRef>` edge from a `StorageBufferNode`.
	 * @return `*this` for chaining.
	 */
	GraphicsNode& add_storage_buffer(graph::DataHandle handle)
	{
		m_storage_buffers.push_back(handle);
		m_inputs.push_back(handle);
		mark_dirty();
		return *this;
	}

	/**
	 * @brief Bind a sampled-image edge into this node's descriptor set.
	 *
	 * `handle` is a `ValueData<gpu::ImageRef>` (e.g. another pass's output). It is sampled by
	 * the shader's `Texture2D` named `name`, via a node-owned sampler that is auto-bound to the
	 * shader's `SamplerState`. The source image must rest in `eShaderReadOnlyOptimal` (its
	 * producer's final layout). A reactive input — its producer is pulled into demand. Call
	 * before adding the node to the graph.
	 *
	 * @param handle `ValueData<gpu::ImageRef>` edge (e.g. another @ref veng::nodes::GraphicsNode's output).
	 * @param name   Reflected `Texture2D` name in the fragment shader.
	 * @return `*this` for chaining.
	 */
	GraphicsNode& add_sampled_image(graph::DataHandle handle, std::string name)
	{
		m_sampled_images.push_back(SampledBinding{.input_index = m_inputs.size(), .name = std::move(name)});
		m_inputs.push_back(handle);
		mark_dirty();
		return *this;
	}

	/**
	 * @brief Rebind the sampled input named `name` to a new source edge at runtime.
	 *
	 * Apply at a frame boundary. Because `inputs()` is read fresh each resolve, repointing away
	 * from a branch makes that branch undemanded — it stops executing, by design, with no
	 * explicit gate.
	 *
	 * @param name   Reflected `Texture2D` name identifying which binding to rebind.
	 * @param handle New `ValueData<gpu::ImageRef>` source edge.
	 */
	void set_sampled_image(std::string_view name, graph::DataHandle handle)
	{
		for (const SampledBinding& binding : m_sampled_images)
		{
			if (binding.name == name)
			{
				m_inputs[binding.input_index] = handle;
				mark_dirty(); // a rebind to a same-revision producer wouldn't dirty us otherwise
				return;
			}
		}
	}

	/**
	 * @brief Set the color the render target is cleared to each frame.
	 *
	 * Default is opaque black `{0, 0, 0, 1}`. Call before adding the node to the graph.
	 *
	 * @param rgba RGBA clear color in linear space.
	 * @return `*this` for chaining.
	 */
	GraphicsNode& clear_color(std::array<float, 4> rgba) noexcept
	{
		m_clear_color = rgba;
		mark_dirty();
		return *this;
	}

	/**
	 * @brief Configure the sampler bound to the shader's `SamplerState`.
	 *
	 * Default is `SamplerConfig::render_target()` — linear, clamp-to-edge, no mips — the
	 * correct default for sampling a full-screen render target. Material textures want
	 * `SamplerConfig::texture()` (repeat, trilinear, anisotropic). Call before the first record
	 * — the sampler is created lazily on first record.
	 *
	 * @param config Sampler configuration to use.
	 * @return `*this` for chaining.
	 */
	GraphicsNode& set_sampler(const gpu::SamplerConfig& config) noexcept
	{
		m_sampler_config = config;
		mark_dirty();
		return *this;
	}

	/**
	 * @brief Set the primitive topology for this pass's pipeline.
	 *
	 * Default is triangle list. Use `rhi::Topology::LINE_LIST` for debug-line
	 * rendering — the consuming mesh's vertex pairs are drawn as line segments. Call before the
	 * first record (the pipeline is built lazily on first record).
	 *
	 * @param topology The desired primitive topology.
	 * @return `*this` for chaining.
	 */
	GraphicsNode& topology(rhi::Topology topology) noexcept
	{
		m_topology = topology;
		mark_dirty();
		return *this;
	}

	/**
	 * @brief Enable or disable depth writes for this pass.
	 *
	 * Depth writes are on by default when a depth format is configured. Disable for debug
	 * overlays that should layer on top without affecting the depth buffer.
	 *
	 * @param enabled `true` to enable depth writes, `false` to disable.
	 * @return `*this` for chaining.
	 */
	GraphicsNode& depth_write(bool enabled) noexcept
	{
		m_depth_write = enabled;
		mark_dirty();
		return *this;
	}

	/**
	 * @brief Request MSAA at the given sample count for this pass (default: `e1`, off).
	 *
	 * The pass renders into a multisampled color + depth attachment and resolves to a
	 * single-sample image that downstream consumers sample/blit unchanged. The request is clamped
	 * to the device's supported framebuffer sample counts on first record, so asking for more than
	 * the hardware offers degrades gracefully. Call before the first record (the pipeline and
	 * targets are configured lazily).
	 *
	 * @param samples The desired sample count.
	 * @return `*this` for chaining.
	 */
	GraphicsNode& samples(rhi::SampleCount samples) noexcept
	{
		m_samples = samples;
		mark_dirty();
		return *this;
	}

	/**
	 * @brief Default-draw shorthand for @ref DrawConfig::set_instances.
	 *
	 * @param count Fixed instance count for the default draw.
	 * @return `*this` for chaining.
	 */
	GraphicsNode& set_instances(std::uint32_t count) noexcept
	{
		default_draw().instance_count = count;
		default_draw().instance_count_source.reset();
		mark_dirty();
		return *this;
	}

	/**
	 * @brief Default-draw shorthand for @ref DrawConfig::set_instances_from.
	 *
	 * @param handle `ValueData<gpu::BufferRef>` edge whose `count` drives `instanceCount`.
	 * @return `*this` for chaining.
	 */
	GraphicsNode& set_instances_from(graph::DataHandle handle)
	{
		default_draw().instance_count_source = handle;
		m_inputs.push_back(handle);
		mark_dirty();
		return *this;
	}

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return m_inputs; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	/**
	 * @brief Return the color target written by the most recent record.
	 *
	 * Returns a @ref veng::ResourcePool physical copy; `nullptr` before the first render. Primarily
	 * a test lens — production code should consume the published `ImageRef` via the graph edge.
	 *
	 * @return Pointer to the last rendered color image, or `nullptr` before first record.
	 */
	[[nodiscard]] const Image* scene() const noexcept { return m_last_color; }

	 protected:
	[[nodiscard]] std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override;
	std::vector<gpu::ImageUsage>						image_usages(graph::ExecContext& ctx) override;

	 private:
	/// Type-erased push-constant binding: stage, byte offset, byte size, and a reader that
	/// resolves the edge to raw bytes at record time.
	struct PushBinding
	{
		rhi::ShaderStage								stage;
		std::uint32_t									offset;
		std::uint32_t									size;
		std::function<const void*(graph::ExecContext&)> read;
	};

	// Build a type-erased push-constant binding: it captures sizeof(T) and a reader that
	// resolves `handle` to a ValueData<T> at record time. Shared by push_constant (first draw)
	// and DrawConfig::push_constant (any draw).
	template <class T>
	static PushBinding make_push(graph::DataHandle handle, rhi::ShaderStage stage, std::uint32_t offset)
	{
		return PushBinding{.stage  = stage,
						   .offset = offset,
						   .size   = static_cast<std::uint32_t>(sizeof(T)),
						   .read   = [handle](graph::ExecContext& ctx) -> const void*
						   {
							   const auto* slot = dynamic_cast<graph::ValueData<T>*>(ctx.data(handle));
							   return slot != nullptr ? static_cast<const void*>(&slot->value()) : nullptr;
						   }};
	}

	/// One draw within the pass: its geometry (a mesh edge, or `vertex_count` `SV_VertexID`
	/// vertices), its push constants, and its instance count. Many draws share the node's
	/// pipeline, render target, uniforms, sampled images and storage buffers.
	/// `instance_count_source` (a `ValueData<gpu::BufferRef>` edge) overrides `instance_count`
	/// when set, reading the count from the published `BufferRef` each record — so the
	/// per-instance storage buffer drives both the data and the draw size.
	struct Draw
	{
		std::optional<graph::DataHandle> mesh{};		   ///< Unset => `SV_VertexID` draw of `vertex_count`.
		std::uint32_t					 vertex_count = 0; ///< `SV_VertexID` fallback vertex count.
		std::uint32_t instance_count = 1; ///< Fixed instance count (overridden by `instance_count_source`).
		std::optional<graph::DataHandle>
								 instance_count_source{}; ///< Set => `instanceCount` read from `BufferRef::count`.
		std::vector<PushBinding> push_constants{};		  ///< Per-draw push-constant bindings.
	};

	// The first draw, created on demand: set_mesh / push_constant configure it, so the
	// single-draw API is just the first entry of the draw list.
	Draw& default_draw()
	{
		if (m_draws.empty())
		{
			m_draws.push_back(Draw{.vertex_count = m_vertex_count});
		}
		return m_draws.front();
	}

	/// A sampled-image input: its current source edge lives at `m_inputs[input_index]` (so a
	/// runtime rebind via @ref set_sampled_image also updates `inputs()`), bound to the
	/// reflected `Texture2D` named `name`.
	struct SampledBinding
	{
		std::size_t input_index; ///< Index into `m_inputs` for the current source edge.
		std::string name;		 ///< Reflected `Texture2D` name in the fragment shader.
	};

	std::string					   m_vertex_shader;
	std::string					   m_fragment_shader;
	rhi::Format					   m_color_format;
	rhi::Format					   m_depth_format; ///< `eUndefined` => no depth attachment.
	std::uint32_t				   m_vertex_count;
	graph::DataHandle			   m_output;
	std::vector<graph::DataHandle> m_inputs; ///< `[screen_size, push-constant + mesh + uniform + sampled edges...]`
	std::vector<Draw> m_draws; ///< One or more draws (see `default_draw()`); per-draw geometry + push constants.
	std::vector<graph::DataHandle>	m_uniforms;		   ///< Descriptor-bound uniform-buffer edges.
	std::vector<graph::DataHandle>	m_storage_buffers; ///< Descriptor-bound SSBO edges (`StructuredBuffer<T>`).
	std::vector<SampledBinding>		m_sampled_images;  ///< Descriptor-bound sampled-image edges.
	std::array<float, 4>			m_clear_color{0.0F, 0.0F, 0.0F, 1.0F};
	rhi::Topology					m_topology	  = rhi::Topology::TRIANGLE_LIST;
	bool							m_depth_write = true; ///< Depth writes enabled (when depth attached).
	std::optional<GraphicsPipeline> m_pipeline;			  ///< Built lazily on first record.

	/// Byte stride of vertex binding 0, reflected from the vertex shader at pipeline build.
	/// Empty when the shader declares no vertex inputs (the `SV_VertexID` fullscreen path).
	/// Each bound `MeshRef::vertex_stride` is validated against this — a mismatch is a typed
	/// node failure, not garbage strided through the buffer.
	std::optional<std::uint32_t> m_expected_vertex_stride;

	/// Descriptor state, populated on first record alongside the pipeline. The set is allocated
	/// and rewritten per frame slot (one set per in-flight frame): a slot's set was last used by
	/// the frame that previously held that slot (now retired), so re-writing it is safe at any
	/// frames-in-flight depth. The sampler is lazily created and auto-bound to every reflected
	/// `SamplerState` binding.
	std::map<std::string, DescriptorInfo>
									   m_descriptors_by_name; ///< Reflected name -> descriptor info (binding + type).
	std::optional<DescriptorAllocator> m_descriptors;
	std::vector<vk::DescriptorSet>	   m_descriptor_sets; ///< One per frame slot, allocated lazily.
	gpu::SamplerConfig m_sampler_config; ///< How the lazy sampler is configured (default: render target).
	rhi::SamplerHandle m_sampler_handle; ///< Device-owned sampler; created lazily when there are sampled images.

	/// Targets live in the engine's @ref veng::ResourcePool (N-buffered) via @ref veng::RenderTargetSet,
	/// not in the node: declared once, a physical copy acquired each record, with the optional MSAA
	/// resolve handled inside the set. `m_last_color` is the (resolved, single-sample) copy written
	/// this record, kept for the @ref scene readback lens. `m_versioned` owns the per-produce version
	/// bump so the published `ImageRef` compares unequal across re-renders (the change-cutoff
	/// signal for consumers).
	rhi::SampleCount	 m_samples = rhi::SampleCount::X1; ///< Requested MSAA level (clamped at build).
	RenderTargetSet		 m_targets;
	const Image*		 m_last_color = nullptr;
	gpu::VersionedOutput m_versioned;
};
} // namespace veng::nodes

#endif // VENG_GRAPHICSNODE_HPP
