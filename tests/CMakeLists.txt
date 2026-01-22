include(CTest)
find_package(Catch2 3 REQUIRED)
include(Catch)

function(add_test_executable TARGET_NAME SRC_FILES)
    target_add_executable(${TARGET_NAME} ${SRC_FILES})
    target_link_libraries(${TARGET_NAME} PRIVATE Catch2::Catch2WithMain)
    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
    catch_discover_tests(${TARGET_NAME})
endfunction()

# VulkanContext tests
add_executable(ContextTests context/VulkanContextTests.cpp)
target_link_libraries(ContextTests PRIVATE veng_context Catch2::Catch2WithMain)


add_executable(ShaderLoadingTests shader/ShaderLoadingTests.cpp)
target_link_libraries(ShaderLoadingTests PRIVATE veng_shader Catch2::Catch2WithMain)
target_compile_options(ShaderLoadingTests PRIVATE -O0)

add_executable(ShaderValidationTests shader/ShaderValidationTests.cpp)
target_link_libraries(ShaderValidationTests PRIVATE veng_shader Catch2::Catch2WithMain)

catch_discover_tests(ContextTests)
catch_discover_tests(ShaderLoadingTests)
catch_discover_tests(ShaderValidationTests)

