// Vertex shader for the debug-line overlay: a `(position, color)` vertex pair per line
// segment, transformed by the camera's view_proj only — no per-instance model matrix, no
// lighting. Pairs with debug_line.frag.
struct VSIn
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 color    : COLOR0;
};

struct VSOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 color : COLOR0;
};

struct PushData
{
    float4x4 view_proj;
};
[[vk::push_constant]] PushData pc;

[shader("vertex")]
VSOut main(VSIn input)
{
    VSOut output;
    output.position = mul(float4(input.position, 1.0), pc.view_proj);
    output.color    = input.color;
    return output;
}
