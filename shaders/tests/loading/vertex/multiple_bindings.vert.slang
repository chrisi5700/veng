struct PerVertex
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
};

struct PerInstance
{
    [[vk::location(2)]] float4 matrixCol0 : INSTANCE0;
    [[vk::location(3)]] float4 matrixCol1 : INSTANCE1;
    [[vk::location(4)]] float4 matrixCol2 : INSTANCE2;
    [[vk::location(5)]] float4 matrixCol3 : INSTANCE3;
};

struct VertexOutput
{
    float4 position : SV_Position;
};

[shader("vertex")]
VertexOutput main(
    PerVertex vertex,           // binding 0
    PerInstance instance,       // binding 1
    uint vertexId : SV_VertexID // skipped - scalar, not from vertex buffer
)
{
    // Reconstruct instance matrix from columns
    float4x4 instanceMatrix = float4x4(
        instance.matrixCol0,
        instance.matrixCol1,
        instance.matrixCol2,
        instance.matrixCol3
    );

    VertexOutput output;
    output.position = mul(instanceMatrix, float4(vertex.position, 1.0));
    return output;
}