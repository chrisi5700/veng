# Add library definitions here
# See README.md for CMake library patterns and examples


add_library(veng_logging INTERFACE)
target_link_libraries(veng_logging
        INTERFACE spdlog::spdlog
)
target_compile_definitions(veng_logging
        INTERFACE
        LOG_FILE="${CMAKE_SOURCE_DIR}/logs/logs.txt"
)
target_include_directories(veng_logging
        INTERFACE
        ${CMAKE_SOURCE_DIR}/include
)

target_add_library(veng_context)
target_sources(veng_context PRIVATE context/Context.cpp)
target_link_libraries(veng_context PUBLIC Vulkan::Vulkan glfw veng_logging)
target_compile_definitions(veng_context PUBLIC
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 # Vulkan specific defines
        VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL=1
        VULKAN_HPP_NO_EXCEPTIONS
        VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
        GLM_FORCE_DEPTH_ZERO_TO_ONE # depth different for vulkan as opposed to opengl
)


target_add_library(veng_shader)
target_sources(veng_shader PRIVATE shader/Shader.cpp)
target_link_libraries(veng_shader PUBLIC veng_context slang::slang glm::glm)
target_compile_definitions(veng_shader PUBLIC
        SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders"
)
