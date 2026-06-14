# rhi_triangle — a node-free, graph-free example that records one triangle straight through the RHI
# (rhi::Device handle registry + rhi::CommandEncoder), for reviewing the RHI surface itself. Unlike
# the other examples it does NOT link veng_example_common (Window/AppLoop) — it links only the engine
# construction layers it sits on: veng_pipelines (GraphicsPipeline + Shader) and veng_resources
# (Image/Buffer), which transitively bring veng_context (the rhi::Device + the RHI headers).

target_add_executable(rhi_triangle main.cpp)
target_compile_features(rhi_triangle PRIVATE cxx_std_26)
target_link_libraries(rhi_triangle PRIVATE veng_pipelines veng_resources)
set_target_properties(rhi_triangle PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "")
