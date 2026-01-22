// Vertex shader with push constants
struct VertexInput
{
    [[vk::location(0)]] float3 position : POSITION;
};

struct VertexOutput
{
    float4 position : SV_Position;
};

struct PushData
{
    float4x4 transform;
    float scale;
};

[[vk::push_constant]]
PushData pushData;

[shader("vertex")]
VertexOutput main(VertexInput input)
{
    VertexOutput output;
    output.position = mul(pushData.transform, float4(input.position * pushData.scale, 1.0));
    return output;
}
