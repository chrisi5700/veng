// Tessellation evaluation (domain) shader
struct DomainInput
{
    [[vk::location(0)]] float3 normal : NORMAL;
    [[vk::location(1)]] float tessLevel : TESS_LEVEL;
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

    // Interpolate normal
    output.normal = normalize(
        patch[0].normal * barycentrics.x +
        patch[1].normal * barycentrics.y +
        patch[2].normal * barycentrics.z);

    output.position = float4(output.normal, 1.0);
    return output;
}
