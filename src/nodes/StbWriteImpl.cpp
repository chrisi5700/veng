/**
 * @file
 * @author chris
 * @brief Single translation unit that compiles the stb_image_write implementation.
 *
 * `stb_image_write` is header-only: exactly one source file must define
 * `STB_IMAGE_WRITE_IMPLEMENTATION` before including the header. @ref veng::nodes::ScreenshotNode
 * includes `<stb_image_write.h>` for declarations only and calls `stbi_write_png`. This TU is built
 * with warnings suppressed (see `src/CMakeLists.txt`) because stb is third-party and not
 * `-Wall/-Werror` clean.
 *
 * @ingroup graph_nodes
 */

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
