// Simple tessellation control shader for testing input/output extraction
struct HullInput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
};

struct HullOutput
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
};

struct PatchConstant
{
    float edges[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
};

PatchConstant PatchConstantFunc(InputPatch<HullInput, 3> patch)
{
    PatchConstant output;
    output.edges[0] = 1.0;
    output.edges[1] = 1.0;
    output.edges[2] = 1.0;
    output.inside = 1.0;
    return output;
}

[shader("hull")]
[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("PatchConstantFunc")]
HullOutput main(
    InputPatch<HullInput, 3> patch,
    uint id : SV_OutputControlPointID)
{
    HullOutput output;
    output.position = patch[id].position;
    output.normal = patch[id].normal;
    return output;
}
