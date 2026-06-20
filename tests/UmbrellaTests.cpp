// Compile-guard for the public umbrella header. If <veng/veng.hpp> (or the host-blit interop helper
// it re-exports) ever stops compiling standalone, this translation unit fails to build. The
// assertions are compile-time only — no Vulkan device is created — so it runs anywhere.

#include <catch2/catch_test_macros.hpp>
#include <veng/veng.hpp>

TEST_CASE("the veng umbrella header compiles and exposes the public API", "[umbrella][api]")
{
	// Touch one public type from each major layer so the include is used, not just preprocessed.
	STATIC_REQUIRE(sizeof(veng::Context) > 0);			   // context
	STATIC_REQUIRE(sizeof(veng::graph::Graph) > 0);		   // reactive render graph
	STATIC_REQUIRE(sizeof(veng::ResourcePool) > 0);		   // resources
	STATIC_REQUIRE(sizeof(veng::gpu::ImageRef) > 0);	   // gpu seam edge value
	STATIC_REQUIRE(sizeof(veng::nodes::GraphicsNode) > 0); // a standard node

	// veng::gpu::blit_into_host_image is an inline function; including the umbrella above already
	// compiles its body, so the host-blit interop helper is covered by this translation unit too.
}
