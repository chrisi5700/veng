// Vertex stage for veng::passes::PhongPass — per-fragment Blinn-Phong with transparency.
//
// Everything the shading needs rides in ONE 160-byte vertex push-constant block (no
// descriptors): the per-object model + the shared camera view_proj + the per-object RGBA
// color + the camera eye (with shininess folded into .w). The fragment stage reads no push
// constants at all — it consumes only interpolated varyings — so the pipeline has a single
// vertex-stage push range and PhongPass can drive the whole pass with pushConstants alone.
//
// Vertex layout is the engine's standard {position, normal, color}, matching the mesh stride
// a MeshNode uploads. The mesh's own vertex color is multiplied by the push color, so a white
// mesh is tinted entirely by the per-object color the caller passes to add_object().
struct VSIn
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;
    [[vk::location(2)]] float3 color    : COLOR0;
};

struct VSOut
{
    float4                     position       : SV_Position;
    [[vk::location(0)]] float3 world_position : POSITION0;
    [[vk::location(1)]] float3 world_normal   : NORMAL0;
    [[vk::location(2)]] float4 color          : COLOR0;  // rgb = tinted, a = opacity
    [[vk::location(3)]] float4 eye            : POSITION1; // xyz = eye, w = shininess
};

struct PushData
{
    float4x4 model;     // offset 0   (64)
    float4x4 view_proj; // offset 64  (64)
    float4   color;     // offset 128 (16) rgb tint, a opacity
    float4   eye;       // offset 144 (16) xyz eye, w shininess
};
[[vk::push_constant]] PushData pc;

[shader("vertex")]
VSOut main(VSIn input)
{
    VSOut output;
    // Slang lays float4x4 row-major; glm is column-major, so apply from the right
    // (mul(v, M) == M * v in math terms) — the convention the rest of the demos use.
    float4 world_pos      = mul(float4(input.position, 1.0), pc.model);
    output.world_position = world_pos.xyz;
    output.position       = mul(world_pos, pc.view_proj);
    output.world_normal   = normalize(mul(float4(input.normal, 0.0), pc.model).xyz);
    output.color          = float4(input.color * pc.color.rgb, pc.color.a);
    output.eye            = pc.eye;
    return output;
}
