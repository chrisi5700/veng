//
// Single translation unit that compiles the stb_image implementation.
//
// stb_image is header-only: exactly one source file in the project must define
// STB_IMAGE_IMPLEMENTATION before including the header. Everywhere else (Texture.cpp) includes
// <stb_image.h> for declarations only. This TU is built with warnings suppressed (see
// src/CMakeLists.txt) because stb is not -Wall/-Werror clean and is third-party.
//

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
