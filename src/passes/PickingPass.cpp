/**
 * @file
 * @author chris
 * @brief Implementation of @ref veng::passes::PickingPass — the @ref veng::passes::PickingReadbackNode private class
 *        (async staging-buffer readback and callback dispatch) and the @ref veng::passes::PickingPass facade.
 * @ingroup render_passes
 */

#include <atomic>
#include <cstdint>
#include <expected>
#include <glm/glm.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <veng/gpu/GpuExecContext.hpp>
#include <veng/gpu/GpuNode.hpp>
#include <veng/gpu/ImageRef.hpp>
#include <veng/gpu/Sink.hpp>
#include <veng/gpu/SubmitContext.hpp>
#include <veng/nodes/GraphicsNode.hpp>
#include <veng/passes/PickingPass.hpp>
#include <veng/rendergraph/data/Data.hpp>
#include <veng/rendergraph/Resolve.hpp>
#include <veng/resources/Buffer.hpp>
#include <veng/resources/ResourcePool.hpp>
#include <veng/rhi/Convert.hpp>

namespace veng::passes
{
// The readback half of the pass. Modelled on ScreenshotNode (copy the source image into a
// host-visible buffer in record, read it in on_retired once the fence has signalled), but it
// decodes a queued set of pixel requests into object ids and fires their callbacks instead of
// writing a file. One staging buffer per in-flight slot, so overlapping picks never read a
// buffer another frame is mid-write into.
class PickingReadbackNode final : public gpu::GpuNode, public gpu::Sink
{
	 public:
	PickingReadbackNode(graph::DataHandle source, graph::DataHandle output) noexcept
		: m_input(source)
		, m_output(output)
	{
		// Starts dirty like any node, so it runs once when first demanded (a harmless early-out
		// in record when there's no pending pick — it just populates the id image). After that it
		// stays clean and out of the plan until a pick arms it (mark_dirty in request), so steady-
		// state picking is free while idle.
	}

	[[nodiscard]] std::span<const graph::DataHandle> inputs() const override { return {&m_input, 1}; }
	[[nodiscard]] std::span<const graph::DataHandle> outputs() const override { return {&m_output, 1}; }

	/// Queue a pick and pull this node into the next plan. Thread-safe against the record/retire
	/// hooks (which run on the frame thread).
	void request(std::uint32_t x, std::uint32_t y, std::function<void(PickResult)> callback)
	{
		{
			const std::scoped_lock lock(m_mutex);
			m_pending.push_back(Request{.x = x, .y = y, .callback = std::move(callback)});
		}
		m_outstanding.fetch_add(1, std::memory_order_relaxed);
		mark_dirty();
	}

	/// True while any queued pick has not yet been decoded + delivered. The driver uses this to
	/// keep pumping full (presenting) frames until the readback retires — see PickingPass::pending.
	[[nodiscard]] bool pending() const noexcept { return m_outstanding.load(std::memory_order_relaxed) > 0; }

	 protected:
	std::vector<gpu::ImageUsage> image_usages(graph::ExecContext& ctx) override
	{
		// Read-side dependency for the executor: copyImageToBuffer needs the source in
		// TRANSFER_SRC. Same contract as ScreenshotNode.
		const auto* src = graph::resolve<gpu::ImageRef>(ctx, m_input);
		if (src == nullptr || src->value().pool_id == gpu::ImageRef::INVALID_POOL_ID)
		{
			return {};
		}
		return {gpu::ImageUsage{.id = src->value().pool_id, .usage = rhi::TextureUsage::TRANSFER_SRC}};
	}

	std::expected<bool, graph::ExecError> record(gpu::GpuExecContext& ctx) override
	{
		const auto* src = graph::resolve<gpu::ImageRef>(ctx, m_input);
		if (src == nullptr)
		{
			return std::unexpected(graph::ExecError::MISSING_INPUT);
		}
		const gpu::ImageRef image = src->value();
		if (!image.texture.valid())
		{
			return false; // the id render hasn't produced anything yet — nothing to capture
		}

		const std::size_t slot = ctx.frame_slot();
		if (m_slots.size() <= slot)
		{
			m_slots.resize(slot + 1);
		}
		if (!m_slots[slot])
		{
			m_slots[slot] = std::make_unique<Slot>();
		}
		Slot& s = *m_slots[slot];

		// Take the requests armed since this slot last ran; decode them in on_retired.
		{
			const std::scoped_lock lock(m_mutex);
			s.in_flight = std::move(m_pending);
			m_pending.clear();
		}
		if (s.in_flight.empty())
		{
			return false; // armed run with nothing to capture (e.g. requests already taken)
		}

		constexpr std::uint32_t bytes_per_pixel = 4; // R8G8B8A8
		const std::uint64_t		size =
			static_cast<std::uint64_t>(image.extent.width) * image.extent.height * bytes_per_pixel;
		if (!s.staging.has_value() || s.staging->size() < size)
		{
			auto buf =
				Buffer::create(ctx.rhi(), size, rhi::BufferUsageFlags::TRANSFER_DST, rhi::MemoryAccess::HOST_VISIBLE);
			if (!buf.has_value() || buf->mapped() == nullptr)
			{
				return std::unexpected(graph::ExecError::NODE_FAILED);
			}
			s.staging = std::move(buf.value());
		}
		s.extent	= image.extent;
		s.row_pitch = image.extent.width * bytes_per_pixel;

		// The id render may be cached (its output unchanged) — retain the pool copy we read while
		// this frame is in flight.
		ctx.pool().consume(image);

		// The source is already in TRANSFER_SRC (transitioned by the executor from image_usages);
		// copy it into host-visible staging and barrier for the host read on_retired performs.
		ctx.encoder().copy_texture_to_host_buffer(image.texture, s.staging->handle(), image.extent);
		return true;
	}

	void on_retired(gpu::SubmitContext& ctx) noexcept override
	{
		const std::size_t slot = ctx.frame_slot();
		if (slot >= m_slots.size() || !m_slots[slot])
		{
			return;
		}
		Slot& s = *m_slots[slot];
		if (s.in_flight.empty() || !s.staging.has_value())
		{
			return;
		}
		// The slot's fence has signalled, so the staging buffer is fully populated and coherent.
		const auto* pixels = static_cast<const std::uint8_t*>(s.staging->mapped());
		for (const Request& req : s.in_flight)
		{
			PickResult result{.hit = false, .id = 0, .x = req.x, .y = req.y};
			if (pixels != nullptr && req.x < s.extent.width && req.y < s.extent.height)
			{
				const std::size_t off =
					(static_cast<std::size_t>(req.y) * s.row_pitch) + (static_cast<std::size_t>(req.x) * 4);
				const std::uint32_t id = static_cast<std::uint32_t>(pixels[off + 0]) |
										 (static_cast<std::uint32_t>(pixels[off + 1]) << 8) |
										 (static_cast<std::uint32_t>(pixels[off + 2]) << 16) |
										 (static_cast<std::uint32_t>(pixels[off + 3]) << 24);
				result.id			   = id;
				result.hit			   = (id != 0);
			}
			if (req.callback)
			{
				req.callback(result);
			}
		}
		m_outstanding.fetch_sub(static_cast<int>(s.in_flight.size()), std::memory_order_relaxed);
		s.in_flight.clear();
	}

	 private:
	struct Request
	{
		std::uint32_t					x;
		std::uint32_t					y;
		std::function<void(PickResult)> callback;
	};

	// Per in-flight slot: its own staging buffer + the requests captured into it, decoded when
	// that slot retires. unique_ptr so growing the vector never has to move a Buffer.
	struct Slot
	{
		std::optional<Buffer> staging;
		rhi::Extent2D		  extent{};
		std::uint32_t		  row_pitch = 0;
		std::vector<Request>  in_flight;
	};

	graph::DataHandle				   m_input;
	graph::DataHandle				   m_output;
	std::mutex						   m_mutex;
	std::vector<Request>			   m_pending; // armed, awaiting capture by the next record
	std::vector<std::unique_ptr<Slot>> m_slots;	  // indexed by frame-in-flight slot
	std::atomic<int> m_outstanding{0};			  // queued-but-not-yet-delivered picks; drives PickingPass::pending
};

PickingPass::PickingPass(graph::Graph& graph, graph::TypedHandle<rhi::Extent2D> screen, rhi::Format depth_format)
	: m_graph(&graph)
{
	using graph::ValueData;
	using nodes::GraphicsNode;

	m_id_image	 = graph.add(std::make_unique<ValueData<gpu::ImageRef>>(gpu::ImageRef{}));
	m_done_token = graph.add(std::make_unique<ValueData<int>>(0));

	// Id render: one draw per object, each a flat id-color, depth-tested so the frontmost wins.
	// R8G8B8A8_UNORM (linear) so the encoded id bytes survive the round-trip exactly; cleared to
	// 0 so background decodes to id 0 (no hit).
	auto render = std::make_unique<GraphicsNode>("passes/picking.vert", "passes/picking.frag", rhi::Format::RGBA8_UNORM,
												 depth_format, 0, screen, m_id_image);
	render->clear_color({0.0F, 0.0F, 0.0F, 0.0F});
	m_render	  = render.get();
	m_render_node = graph.add(std::move(render));
	graph.set_producer(m_id_image, m_render_node);

	// Readback: copies the id image to host memory and decodes queued picks on retirement.
	auto readback	= std::make_unique<PickingReadbackNode>(m_id_image, m_done_token);
	m_readback		= readback.get();
	m_readback_node = graph.add(std::move(readback));
	graph.set_producer(m_done_token, m_readback_node);
}

void PickingPass::add_object(graph::DataHandle mesh, graph::DataHandle mvp, std::uint32_t id)
{
	// id as a constant source for the push constant (offset 64, after the mat4 mvp at 0).
	const graph::TypedHandle<std::uint32_t> id_src = m_graph->add_source<std::uint32_t>(id);
	m_render->add_draw(mesh)
		.push_constant<glm::mat4>(mvp, rhi::ShaderStage::VERTEX, 0)
		.push_constant<std::uint32_t>(id_src, rhi::ShaderStage::VERTEX, 64);
}

void PickingPass::pick(std::uint32_t x, std::uint32_t y, std::function<void(PickResult)> on_result)
{
	m_readback->request(x, y, std::move(on_result));
}

bool PickingPass::pending() const noexcept
{
	return m_readback->pending();
}
} // namespace veng::passes
