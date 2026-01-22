// Fragment shader matching geometry output
struct FragmentInput
{
    [[vk::location(0)]] float4 color : COLOR;
    [[vk::location(1)]] float3 barycentrics : TEXCOORD;
};

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target
{
    return input.color * input.barycentrics.x;
}
