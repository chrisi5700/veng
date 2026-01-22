// Fragment shader with texture sampling
struct FragmentInput
{
    [[vk::location(0)]] float2 texCoord : TEXCOORD;
};

[[vk::binding(0, 0)]]
Texture2D albedoTexture;

[[vk::binding(1, 0)]]
SamplerState textureSampler;

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target
{
    return albedoTexture.Sample(textureSampler, input.texCoord);
}
