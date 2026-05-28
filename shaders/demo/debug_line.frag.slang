// Fragment shader for the debug-line overlay: pass through the interpolated per-vertex
// color, no lighting (debug lines should look the same regardless of orientation). Pairs
// with debug_line.vert.
struct PSIn
{
    [[vk::location(0)]] float3 color : COLOR0;
};

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    return float4(input.color, 1.0);
}
