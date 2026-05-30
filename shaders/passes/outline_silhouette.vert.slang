// Silhouette vertex shader for veng::passes::OutlinePass. Transforms buffer-backed geometry
// by a per-object MVP push constant and forwards color to the (white-mask) fragment shader.
//
// The vertex input mirrors the engine's standard mesh layout {position, normal, color} (the
// same one shaders/demo/lit_instanced.vert and the example Geometry library use) so the
// reflected vertex stride matches the mesh buffer. Only `position` is read here, but `normal`
// and `color` MUST be declared anyway: the pipeline's vertex stride comes from the shader's
// reflection (GraphicsPipeline: "vertex input straight from the vertex shader's reflection"),
// so dropping them would shrink the stride and misread every vertex past the first.
struct VSIn
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;  // unused; declared to keep the stride right
    [[vk::location(2)]] float3 color    : COLOR0;  // unused (mask is solid white)
};

struct VSOut
{
    float4              position : SV_Position;
    [[vk::location(0)]] float3 color : COLOR0;
};

struct PushData
{
    float4x4 mvp;
};
[[vk::push_constant]] PushData pc;

[shader("vertex")]
VSOut main(VSIn input)
{
    VSOut output;
    // The column-major glm matrix arrives transposed (Slang lays the float4x4 push constant
    // out row-major), so apply it from the right to get M * v — the engine's mesh.vert
    // convention.
    output.position = mul(float4(input.position, 1.0), pc.mvp);
    output.color    = input.color;
    return output;
}
