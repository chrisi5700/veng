// Per-face flat color, interpolated from the cube vertex shader's COLOR0 output.
struct PSIn
{
    [[vk::location(0)]] float3 color : COLOR0;
};

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    return float4(input.color, 1.0);
}
