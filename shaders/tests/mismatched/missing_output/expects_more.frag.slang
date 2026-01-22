// Fragment shader expecting both normal AND texCoord
struct FragmentInput
{
    [[vk::location(0)]] float3 normal : NORMAL;
    [[vk::location(1)]] float2 texCoord : TEXCOORD;
};

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target
{
    return float4(input.normal, input.texCoord.x);
}
