// Tessellation control (hull) shader
struct HullInput
{
    [[vk::location(0)]] float3 normal : NORMAL;
};

struct HullOutput
{
    [[vk::location(0)]] float3 normal : NORMAL;
    [[vk::location(1)]] float tessLevel : TESS_LEVEL;
};

struct PatchConstant
{
    float edges[3] : SV_TessFactor;
    float inside : SV_InsideTessFactor;
};

PatchConstant PatchConstantFunc(
    InputPatch<HullInput, 3> patch)
{
    PatchConstant output;
    output.edges[0] = 4.0;
    output.edges[1] = 4.0;
    output.edges[2] = 4.0;
    output.inside = 4.0;
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
    output.normal = patch[id].normal;
    output.tessLevel = 4.0;
    return output;
}
