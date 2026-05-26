// Outputs the interpolated per-vertex color, so a readback can prove the color came from
// the vertex buffer (not a hardcoded constant like triangle.frag).
struct PSIn
{
    [[vk::location(0)]] float3 color : COLOR0;
};

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    return float4(input.color, 1.0);
}
