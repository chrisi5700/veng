// Fragment shader for tessellation pipeline
struct FragmentInput
{
    [[vk::location(0)]] float3 normal : NORMAL;
};

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target
{
    return float4(normalize(input.normal) * 0.5 + 0.5, 1.0);
}
