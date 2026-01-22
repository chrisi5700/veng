// Vertex shader with descriptor array for exact size testing
struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
};

struct VertexOutput
{
    float4 position : SV_Position;
};

struct MVP
{
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};

// Array of 4 MVP uniform buffers
[[vk::binding(0, 0)]]
ConstantBuffer<MVP> mvps[4];

[shader("vertex")]
VertexOutput main(VertexInput input, uint instanceID : SV_InstanceID)
{
    VertexOutput output;
    uint idx = min(instanceID, 3);
    output.position = mul(mvps[idx].proj,
                         mul(mvps[idx].view,
                            mul(mvps[idx].model, float4(input.position, 1.0))));
    return output;
}
