// Simple vertex shader with uniform buffer
struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
};

struct VertexOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 normal : NORMAL;
};

struct Transforms
{
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Transforms> transforms;

[shader("vertex")]
VertexOutput main(VertexInput input)
{
    VertexOutput output;
    float4 worldPos = mul(transforms.model, float4(input.position, 1.0));
    float4 viewPos = mul(transforms.view, worldPos);
    output.position = mul(transforms.proj, viewPos);
    output.normal = input.normal;
    return output;
}
