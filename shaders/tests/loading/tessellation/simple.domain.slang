// Simple tessellation evaluation shader for testing input/output extraction
struct DomainInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
};

struct DomainOutput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 normal : NORMAL;
};

struct PatchConstant
{
    float edges[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
};

[shader("domain")]
[domain("tri")]
DomainOutput main(
    PatchConstant patchConst,
    float3 barycentrics : SV_DomainLocation,
    OutputPatch<DomainInput, 3> patch)
{
    DomainOutput output;
    output.position = float4(
        patch[0].position * barycentrics.x +
        patch[1].position * barycentrics.y +
        patch[2].position * barycentrics.z,
        1.0);
    output.normal = normalize(
        patch[0].normal * barycentrics.x +
        patch[1].normal * barycentrics.y +
        patch[2].normal * barycentrics.z);
    return output;
}
