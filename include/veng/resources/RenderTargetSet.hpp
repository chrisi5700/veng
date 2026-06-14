/**
 * @file
 * @author chris
 * @brief Pool-backed color (+ optional depth) render target with optional MSAA resolve.
 *
 * Every dynamic-rendering node in the engine (the generic @ref veng::nodes::GraphicsNode and the
 * @ref veng::passes::PbrPass / @ref veng::passes::PhongPass render nodes) used to hand-roll the
 * same sequence: declare a color + depth image on the @ref veng::ResourcePool, acquire this
 * frame's copies, transition them to their attachment layouts, and `beginRendering` with one
 * color attachment + an optional depth attachment. That block was copy-pasted three times, so
 * adding MSAA would have meant writing the resolve wiring three times too.
 *
 * @ref veng::RenderTargetSet folds that lifecycle into one place. Configured with a sample count,
 * it transparently switches between two layouts:
 *
 * - **Single-sample** (`samples == e1`): one color image, rendered into and published directly.
 * - **MSAA** (`samples > e1`): a multisampled color attachment and a multisampled depth, with the
 *   color attachment resolved into a single-sample image at end-of-pass. The *resolved* image is
 *   what `color()` returns and what a node publishes downstream, so the node's publish + the
 *   executor's post-pass layout transition still key off the same single-sample id in both modes —
 *   MSAA stays invisible past the resolve.
 *
 * @ingroup resources
 */

#ifndef VENG_RENDERTARGETSET_HPP
#define VENG_RENDERTARGETSET_HPP

#include <array>
#include <expected>
#include <veng/resources/Image.hpp>
#include <veng/resources/ResourcePool.hpp>
#include <veng/rhi/CommandEncoder.hpp>
#include <vulkan/vulkan.hpp>

namespace veng
{
class Context;

/**
 * @brief Clamp a requested MSAA sample count to what the device supports for framebuffers.
 *
 * Returns the highest power-of-two sample count `<= requested` that the device reports for both
 * color and depth framebuffer attachments (`limits.framebufferColorSampleCounts` &
 * `framebufferDepthSampleCounts`), or `e1` if none qualifies. Use this before configuring a
 * @ref RenderTargetSet so a caller asking for `e8` on a device that maxes out at `e4` degrades
 * gracefully instead of triggering a validation error.
 *
 * @param ctx       Engine context (queried for physical-device limits).
 * @param requested The desired sample count.
 * @return The clamped, device-supported sample count.
 */
[[nodiscard]] vk::SampleCountFlagBits clamp_sample_count(const Context&			 ctx,
														 vk::SampleCountFlagBits requested) noexcept;

/**
 * @brief A pool-backed color (+ optional depth) render target, optionally multisampled.
 *
 * Holds the logical pool ids and per-frame physical copies for a render pass's attachments and
 * drives them through declare → acquire → begin each frame. One instance lives per render node.
 *
 * @ingroup resources
 * @see ResourcePool
 * @see clamp_sample_count
 */
class RenderTargetSet
{
	 public:
	/**
	 * @brief Configure the target's formats and sample count.
	 *
	 * Call once before the first @ref acquire (typically when the owning node first records, with
	 * an already-clamped sample count). `depth_format == eUndefined` renders without a depth
	 * attachment. The clear color is supplied per frame to @ref begin, so a node may vary it.
	 *
	 * @param color_format Color attachment format.
	 * @param depth_format Depth attachment format, or `eUndefined` for none.
	 * @param samples      Sample count; `e1` disables MSAA.
	 */
	void configure(vk::Format color_format, vk::Format depth_format, vk::SampleCountFlagBits samples) noexcept;

	/// @return The configured sample count.
	[[nodiscard]] vk::SampleCountFlagBits sample_count() const noexcept { return m_samples; }
	/// @return Whether MSAA is enabled (sample count > `e1`).
	[[nodiscard]] bool multisampled() const noexcept { return m_samples != vk::SampleCountFlagBits::e1; }
	/// @return Whether a depth attachment is configured.
	[[nodiscard]] bool has_depth() const noexcept { return m_depth_format != vk::Format::eUndefined; }

	/**
	 * @brief Declare the logical resources (lazily, once) and acquire this frame's copies.
	 *
	 * After this returns successfully, @ref color() / @ref color_id() are valid for this frame.
	 *
	 * @param pool   The engine resource pool.
	 * @param extent This frame's render extent (a change reallocates the copies).
	 * @return Nothing on success, or the `vk::Result` of the first failed image acquire.
	 */
	[[nodiscard]] std::expected<void, vk::Result> acquire(ResourcePool& pool, vk::Extent2D extent);

	/**
	 * @brief Transition the attachments to their attachment-optimal layouts and begin rendering.
	 *
	 * Issues one color attachment (MSAA-resolved into @ref color() when multisampled) and an
	 * optional depth attachment, then calls `beginRendering`. The caller records draws and must
	 * call `cmd.endRendering()` itself. Must be preceded by a successful @ref acquire this frame.
	 *
	 * @param pool        The engine resource pool (records the auto-tracked layout transitions).
	 * @param cmd         The command buffer to record into.
	 * @param extent      This frame's render extent (the render area).
	 * @param clear_color RGBA color the attachment is cleared to this frame.
	 */
	void begin(ResourcePool& pool, rhi::CommandEncoder& enc, vk::Extent2D extent, std::array<float, 4> clear_color);

	/// @return The single-sample image a consumer samples/blits — the resolve target under MSAA,
	///         the color attachment otherwise. Valid after @ref acquire; null before.
	[[nodiscard]] Image* color() const noexcept { return m_color; }
	/// @return The pool id of the single-sample image @ref color() returns (for publish + barriers).
	[[nodiscard]] ImageId color_id() const noexcept { return m_color_id; }

	 private:
	vk::Format				m_color_format = vk::Format::eUndefined;
	vk::Format				m_depth_format = vk::Format::eUndefined;
	vk::SampleCountFlagBits m_samples	   = vk::SampleCountFlagBits::e1;

	bool	m_declared = false;
	ImageId m_color_id = 0; ///< Single-sample image: rendered-into (no MSAA) or resolve target (MSAA).
	ImageId m_msaa_id  = 0; ///< Multisampled color attachment; valid only when @ref multisampled().
	ImageId m_depth_id = 0; ///< Depth attachment (sample count == m_samples); valid only with depth.

	Image* m_color = nullptr; ///< Acquired copy of m_color_id this frame.
	Image* m_msaa  = nullptr; ///< Acquired copy of m_msaa_id this frame (MSAA only).
	Image* m_depth = nullptr; ///< Acquired copy of m_depth_id this frame (with depth).
};
} // namespace veng

#endif // VENG_RENDERTARGETSET_HPP
