cmake_minimum_required(VERSION 3.28)
project(veng)

# Include CMake modules for target helpers and compiler settings
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(TargetHelpers)

# Default cache variables (can be overridden by CMakePresets.json or command line)
set(ENABLE_COMPILER_WARNINGS ON CACHE BOOL "Enable compiler warnings")
set(TREAT_WARNINGS_AS_ERRORS OFF CACHE BOOL "Treat warnings as errors")
set(ENABLE_SANITIZER_ADDRESS OFF CACHE BOOL "Enable AddressSanitizer")
set(ENABLE_SANITIZER_THREAD OFF CACHE BOOL "Enable ThreadSanitizer")
set(ENABLE_SANITIZER_UNDEFINED OFF CACHE BOOL "Enable UndefinedBehaviorSanitizer")
set(ENABLE_LTO OFF CACHE BOOL "Enable Link Time Optimization")

find_package(Threads REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(slang REQUIRED)
find_package(vk-bootstrap CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(unofficial-vulkan-memory-allocator-hpp CONFIG REQUIRED)

# Enable testing at the top level so `ctest` run from the root build dir recurses
# into tests/ (catch_discover_tests registers there). Without this the root
# CTestTestfile is empty and `ctest -N` reports zero tests.
enable_testing()

add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(bench)
add_subdirectory(tests)
# add_subdirectory(libs/SFML) # for example to add other cmake dependencies