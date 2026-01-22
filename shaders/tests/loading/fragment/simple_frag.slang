// Simple fragment shader
struct FragmentInput
{
    [[vk::location(0)]] float3 normal : NORMAL;
};

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target
{
    float3 normalized = normalize(input.normal);
    return float4(normalized * 0.5 + 0.5, 1.0);
}
