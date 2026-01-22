// Fragment shader with multiple descriptor types
struct FragmentInput
{
    [[vk::location(0)]] float2 texCoord : TEXCOORD;
};

struct MaterialData
{
    float4 albedo;
    float metallic;
    float roughness;
};

[[vk::binding(0, 0)]]
ConstantBuffer<MaterialData> material;

[[vk::binding(1, 0)]]
Texture2D albedoMap;

[[vk::binding(2, 0)]]
Texture2D normalMap;

[[vk::binding(3, 0)]]
SamplerState textureSampler;

[shader("fragment")]
float4 main(FragmentInput input) : SV_Target
{
    float4 albedo = albedoMap.Sample(textureSampler, input.texCoord);
    float3 normal = normalMap.Sample(textureSampler, input.texCoord).xyz;
    return albedo * material.albedo;
}
