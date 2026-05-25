
target_add_executable(playground main.cpp)
target_compile_features(playground PRIVATE cxx_std_26)
target_link_libraries(playground PRIVATE veng_render_graph)
# target_compile_definitions(playground PRIVATE SOME_DEFINE=VALUE)