//
// Created by chris on 5/26/26.
//
// L4 node — a *generic* raster node (design.md §L4). It is self-contained: you give it
// the names of a vertex + fragment shader, the attachment formats, a screen-size input
// that sizes its owned color (+ optional depth) target, and zero or more push-constant
// value edges. It builds its own pipeline internally (the first time it records, when it
// has a device) — graph construction never touches the L2 pipeline layer — then draws and
// publishes the rendered target as an `ImageData`. "The cube" is just this node wired to
// the cube shaders plus an `mvp` push-constant edge fed by a pure transform of the angle:
// no cube-specific node. A resize reallocates the target (an ordinary invalidation); a
// changed push-constant re-renders, an unchanged one is cached.
//
// Geometry is either buffer-backed — `set_mesh(handle)` binds a `gpu::MeshRef` produced by
// a MeshNode and the node draws it (indexed when the ref has an index buffer) — or, with no
// mesh bound, a `vertex_count` draw of vertices fabricated in the shader from SV_VertexID (a
// fullscreen post-process triangle is just `vertex_count = 3` + a fullscreen vertex shader).
// One node can host *several* draws via `add_draw(mesh)` — they share its pipeline, target,
// uniforms and sampled images, so N meshes land in one pass (a scene), each with its own push
// constants (e.g. a per-object model matrix). `set_mesh`/`push_constant` are the single-draw
// shorthand for the first such draw.
// `add_uniform(handle)` binds a `gpu::UniformRef` (from a UniformNode) into the node's own
// descriptor set, matched to the shader's reflected binding by name. `add_sampled_image(
// handle, name)` binds a `gpu::ImageRef` as a sampled texture (matched the same way, with a
// node-owned sampler) — this is what lets one pass read another's output, the basis of
// multi-pass effects. `set_sampled_image(name, handle)` rebinds such an input at runtime: a
// rebind that drops an upstream branch makes that branch undemanded, so it stops executing.
// `final_layout(...)` chooses the layout the output rests in: `eTransferSrcOptimal` (default,
// for a blit/readback) or `eShaderReadOnlyOptimal` (for an output another pass will sample).
//

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
#include <veng/pipelines/GraphicsPipeline.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/RenderGraphCommon.hpp>
#include <veng/resources/Image.hpp>
#include <veng/shader/Shader.hpp>
#include <vulkan/vulkan.hpp>

namespace veng::nodes
{
class GraphicsNode final : public gpu::GpuNode
{
	 public:
	/// `vertex_shader`/`fragment_shader` are shader names resolved from SHADER_DIR; the
	/// node loads them and builds its pipeline for (`color_format`[, `depth_format`]) on
	/// first record. Pass `eUndefined` for `depth_format` to render without depth.
	/// `screen_size` is a `ValueData<vk::Extent2D>` source that sizes the owned target;
	/// `output` is the scene `ImageData` (left in TRANSFER_SRC). `vertex_count` is the draw
	/// count (geometry comes from the vertex shader for now).
	GraphicsNode(std::string vertex_shader, std::string fragment_shader, vk::Format color_format,
				 vk::Format depth_format, std::uint32_t vertex_count, graph::DataHandle screen_size,
				 graph::DataHandle output) noexcept;

	// Owns a raw vk::Sampler, so it is non-copyable and (being held in a unique_ptr and never
	// moved) non-movable; the destructor frees the sampler.
	GraphicsNode(const GraphicsNode&)			 = delete;
	GraphicsNode& operator=(const GraphicsNode&) = delete;
	GraphicsNode(GraphicsNode&&)				 = delete;
	GraphicsNode& operator=(GraphicsNode&&)		 = delete;
	~GraphicsNode();

	/// Bind a push-constant value edge. `T` is the value type on `handle`; its bytes are
	/// pushed at `offset` for `stage` each time the node records. The edge is a reactive
	/// input — changing it re-renders the node. Returns *this for chaining; call before
	/// adding the node to the graph (it extends the input set).
	template <class T>
	GraphicsNode& push_constant(graph::DataHandle handle, vk::ShaderStageFlags stage = vk::ShaderStageFlagBits::eVertex,
								std::uint32_t offset = 0)
	{
		default_draw().push_constants.push_back(make_push<T>(handle, stage, offset));
		m_inputs.push_back(handle); // a push-constant edge is a dirtiness input
		mark_dirty();				// extending m_inputs at runtime must re-plan us
		return *this;
	}

	/// Bind a mesh edge: the node draws the buffer-backed geometry on `handle` (a
	/// `ValueData<gpu::MeshRef>`, produced by a MeshNode) instead of fabricating vertices
	/// from SV_VertexID. Indexed if the ref carries an index buffer, else a non-indexed draw
	/// of the ref's vertex count — either way overriding the constructor's `vertex_count`.
	/// The vertex layout must match the vertex shader's reflected input. A reactive input
	/// (re-uploading the mesh re-renders). Returns *this for chaining; call before adding the
	/// node to the graph (it extends the input set).
	GraphicsNode& set_mesh(graph::DataHandle handle)
	{
		default_draw().mesh = handle;
		m_inputs.push_back(handle);
		mark_dirty();
		return *this;
	}

	/// Per-draw configuration returned by `add_draw`: attach push-constant edges to that one
	/// draw (e.g. a per-object model matrix). Pass-level state (uniforms, sampled images, the
	/// target, clear color, final layout) lives on the node and is shared by every draw.
	class DrawConfig
	{
		 public:
		/// Bind a push-constant edge to this draw only (see GraphicsNode::push_constant). The
		/// edge is a reactive input — changing it re-renders the pass.
		template <class T>
		DrawConfig& push_constant(graph::DataHandle	   handle,
								  vk::ShaderStageFlags stage  = vk::ShaderStageFlagBits::eVertex,
								  std::uint32_t		   offset = 0)
		{
			m_node->m_draws[m_index].push_constants.push_back(GraphicsNode::make_push<T>(handle, stage, offset));
			m_node->m_inputs.push_back(handle);
			m_node->mark_dirty();
			return *this;
		}

		/// Set this draw's `instanceCount` to a fixed value (default 1). The shader reads
		/// `SV_InstanceID` to know which instance it is on; for a non-instanced draw leave it
		/// unset.
		DrawConfig& set_instances(std::uint32_t count) noexcept
		{
			m_node->m_draws[m_index].instance_count = count;
			m_node->m_draws[m_index].instance_count_source.reset();
			m_node->mark_dirty();
			return *this;
		}

		/// Drive this draw's `instanceCount` reactively from a `ValueData<gpu::BufferRef>`
		/// edge — typically the same storage buffer that supplies the per-instance data. Each
		/// record reads `BufferRef::count` to decide how many instances to emit; a resize of
		/// the source array is therefore a single producer write that drives both the upload
		/// and the draw size. Adds the edge as a reactive input so a count change re-runs the
		/// draw.
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

	/// Append a draw to this pass. With a valid `mesh` (a `ValueData<gpu::MeshRef>`, from a
	/// MeshNode) it draws that buffer-backed geometry (indexed when the ref carries indices);
	/// otherwise it draws `vertex_count` SV_VertexID vertices. Returns a DrawConfig to attach
	/// this draw's push constants. Every draw shares the node's pipeline, target, uniforms and
	/// sampled images — so this is how N meshes render into one pass, each with its own
	/// transform. Call before adding the node to the graph (it extends the input set).
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

	/// Bind a uniform edge: `handle` (a `ValueData<gpu::UniformRef>`, produced by a
	/// UniformNode) is written into this node's descriptor set at the binding whose reflected
	/// name matches the ref's `name`. A reactive input — changing the uniform value re-renders
	/// the node. Returns *this for chaining; call before adding the node to the graph (it
	/// extends the input set).
	GraphicsNode& add_uniform(graph::DataHandle handle)
	{
		m_uniforms.push_back(handle);
		m_inputs.push_back(handle);
		mark_dirty();
		return *this;
	}

	/// Bind a storage-buffer edge: `handle` (a `ValueData<gpu::BufferRef>`, produced by a
	/// StorageBufferNode) is written into this node's descriptor set at the binding whose
	/// reflected name matches the ref's `name` — the shader sees it as a `StructuredBuffer<T>`
	/// (or `RWStructuredBuffer<T>` for the storage-image equivalent). A reactive input —
	/// re-uploading the array re-renders the node. The published ref's `count` is what the
	/// `set_instances_from(handle)` shorthand reads to drive `instanceCount`, so the same edge
	/// can also configure the draw size. Returns *this for chaining; call before adding the
	/// node to the graph (it extends the input set).
	GraphicsNode& add_storage_buffer(graph::DataHandle handle)
	{
		m_storage_buffers.push_back(handle);
		m_inputs.push_back(handle);
		mark_dirty();
		return *this;
	}

	/// Bind a sampled-image edge: `handle` (a `ValueData<gpu::ImageRef>`, e.g. another pass's
	/// output) is sampled by the shader's `Texture2D` named `name`, via a node-owned sampler
	/// auto-bound to the shader's `SamplerState`. The source image must rest in
	/// `eShaderReadOnlyOptimal` (its producer's `final_layout`). A reactive input — its
	/// producer is pulled into demand. Returns *this; call before adding the node to the graph.
	GraphicsNode& add_sampled_image(graph::DataHandle handle, std::string name)
	{
		m_sampled_images.push_back(SampledBinding{.input_index = m_inputs.size(), .name = std::move(name)});
		m_inputs.push_back(handle);
		mark_dirty();
		return *this;
	}

	/// Rebind the sampled input named `name` to a new source edge at runtime (apply at a frame
	/// boundary). Because `inputs()` is read fresh each resolve, repointing away from a branch
	/// makes that branch undemanded — it stops executing, by design, with no explicit gate.
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

	/// Set the color the target is cleared to each frame (RGBA, default opaque black).
	/// Returns *this for chaining; call before adding the node to the graph.
	GraphicsNode& clear_color(std::array<float, 4> rgba) noexcept
	{
		m_clear_color = rgba;
		mark_dirty();
		return *this;
	}

	/// Set the primitive topology for this pass's pipeline (default triangle list). Use
	/// `vk::PrimitiveTopology::eLineList` for debug-line rendering — the consuming mesh's
	/// vertex pairs are then drawn as line segments. Call before the first record (the
	/// pipeline is built lazily on first record).
	GraphicsNode& topology(vk::PrimitiveTopology topology) noexcept
	{
		m_topology = topology;
		mark_dirty();
		return *this;
	}

	/// Disable depth writes for this pass (writes are on by default when a depth format is
	/// configured). Use for debug overlays you want layered on top without affecting depth.
	GraphicsNode& depth_write(bool enabled) noexcept
	{
		m_depth_write = enabled;
		mark_dirty();
		return *this;
	}

	/// Default-draw shorthand for `DrawConfig::set_instances` — see that method.
	GraphicsNode& set_instances(std::uint32_t count) noexcept
	{
		default_draw().instance_count = count;
		default_draw().instance_count_source.reset();
		mark_dirty();
		return *this;
	}

	/// Default-draw shorthand for `DrawConfig::set_instances_from` — see that method.
	GraphicsNode& set_instances_from(graph::DataHandle handle)
	{
		default_draw().instance_count_source = handle;
		m_inputs.push_back(handle);
		mark_dirty();
		return *this;
	}

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return m_inputs; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	/// The color target written by the most recent record (a ResourcePool copy), or nullptr
	/// before the first render (test lens).
	[[nodiscard]] const Image* scene() const noexcept { return m_last_color; }

	 protected:
	[[nodiscard]] std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override;
	std::vector<gpu::ImageUsage>						image_usages(graph::ExecContext& ctx) override;

	 private:
	struct PushBinding
	{
		vk::ShaderStageFlags							stage;
		std::uint32_t									offset;
		std::uint32_t									size;
		std::function<const void*(graph::ExecContext&)> read;
	};

	// Build a type-erased push-constant binding: it captures sizeof(T) and a reader that
	// resolves `handle` to a ValueData<T> at record time. Shared by push_constant (first draw)
	// and DrawConfig::push_constant (any draw).
	template <class T>
	static PushBinding make_push(graph::DataHandle handle, vk::ShaderStageFlags stage, std::uint32_t offset)
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

	// One draw within the pass: its geometry (a mesh edge, or `vertex_count` SV_VertexID
	// vertices), its push constants, and its instance count. Many draws share the node's
	// pipeline, target, uniforms, sampled images and storage buffers. `instance_count_source`
	// (a `ValueData<gpu::BufferRef>` edge) overrides `instance_count` when set, reading the
	// count from the published BufferRef each record — so the per-instance storage buffer
	// drives both the data and the draw size.
	struct Draw
	{
		std::optional<graph::DataHandle> mesh{}; // unset => SV_VertexID draw of vertex_count
		std::uint32_t					 vertex_count	= 0;
		std::uint32_t					 instance_count = 1;
		std::optional<graph::DataHandle> instance_count_source{}; // set => count from BufferRef
		std::vector<PushBinding>		 push_constants{};
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

	// A sampled-image input: its current source edge lives at m_inputs[input_index] (so a
	// runtime rebind updates inputs() too), bound to the reflected Texture2D named `name`.
	struct SampledBinding
	{
		std::size_t input_index;
		std::string name;
	};

	std::string						m_vertex_shader;
	std::string						m_fragment_shader;
	vk::Format						m_color_format;
	vk::Format						m_depth_format; // eUndefined => no depth attachment
	std::uint32_t					m_vertex_count;
	graph::DataHandle				m_output;
	std::vector<graph::DataHandle>	m_inputs;	// [screen_size, push-constant + mesh + uniform + sampled edges...]
	std::vector<Draw>				m_draws;	// >=1 draw (see default_draw): per-draw geometry + push constants
	std::vector<graph::DataHandle>	m_uniforms; // descriptor-bound uniform-buffer edges
	std::vector<graph::DataHandle>	m_storage_buffers; // descriptor-bound SSBO edges (StructuredBuffer<T>)
	std::vector<SampledBinding>		m_sampled_images;  // descriptor-bound sampled-image edges
	std::array<float, 4>			m_clear_color{0.0F, 0.0F, 0.0F, 1.0F};
	vk::PrimitiveTopology			m_topology	  = vk::PrimitiveTopology::eTriangleList;
	bool							m_depth_write = true; // depth writes (when depth attached)
	std::optional<GraphicsPipeline> m_pipeline;			  // built lazily on first record

	// Descriptor state, populated on first record (alongside the pipeline). The set is allocated
	// and rewritten PER frame slot (one set per in-flight frame): a slot's set was last used by
	// the frame that previously held that slot, which has retired, so re-writing it is safe at any
	// frames-in-flight. The sampler is lazily created and auto-bound to every reflected
	// SamplerState binding.
	std::map<std::string, DescriptorInfo> m_descriptors_by_name; // reflected name -> info (binding + type)
	std::optional<DescriptorAllocator>	  m_descriptors;
	std::vector<vk::DescriptorSet>		  m_descriptor_sets; // one per frame slot, allocated lazily
	vk::Sampler							  m_sampler;		 // lazily created when there are sampled images
	vk::Device							  m_device;			 // captured to free m_sampler in the destructor

	// Targets live in the engine's ResourcePool (N-buffered), not in the node: declared once,
	// a physical copy acquired each record. m_last_color is the copy written this record, kept
	// for the scene() readback lens. m_version is bumped on every produce so the published
	// ImageRef compares unequal across re-renders (the change-cutoff signal for consumers).
	bool		  m_declared   = false;
	ImageId		  m_color_id   = 0;
	ImageId		  m_depth_id   = 0;
	const Image*  m_last_color = nullptr;
	std::uint64_t m_version	   = 0;
};
} // namespace veng::nodes

#endif // VENG_GRAPHICSNODE_HPP
