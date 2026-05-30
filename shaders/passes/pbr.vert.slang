// Vertex stage for veng::passes::PbrPass — metallic-roughness PBR (review.md item 5).
//
// Vertex layout is the engine's standard PBR vertex {position, normal, tangent, uv} (48 bytes,
// gpu::PbrVertex). The shared camera/light live in a Frame uniform (set 0, binding 0, read by
// both stages); the per-object model + material factors ride in a 112-byte push block. We emit
// world-space position / normal / tangent and the UV for the fragment stage to light and texture.

struct VSIn
{
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal   : NORMAL;
    [[vk::location(2)]] float4 tangent  : TANGENT; // xyz + handedness in .w
    [[vk::location(3)]] float2 uv       : TEXCOORD0;
};

struct VSOut
{
    float4                     position       : SV_Position;
    [[vk::location(0)]] float3 world_position : POSITION0;
    [[vk::location(1)]] float3 world_normal   : NORMAL0;
    [[vk::location(2)]] float4 world_tangent  : TANGENT0; // xyz world tangent, w handedness
    [[vk::location(3)]] float2 uv             : TEXCOORD0;
};

struct Frame
{
    float4x4 view_proj;      // 0
    float4x4 view;           // 64  world->view, for the fragment's clustered depth slice
    float4   camera_pos;     // 128 xyz
    float4   light_dir;      // 144 xyz = unit direction from surface TO the light
    float4   light_color;    // 160 rgb = colour * intensity
    float4   ambient;        // 176 rgb
    uint4    cluster_dims;   // 192 xyz = froxel grid dims, w = cluster count (0 => no clustered lights)
    float4   cluster_params; // 208 x z_near, y z_far, zw = screen extent (px)
};
[[vk::binding(0, 0)]] ConstantBuffer<Frame> frame;

struct Push
{
    float4x4 model;      // 0   per-object world transform
    float4   base_color; // 64  baseColorFactor (rgba)
    float4   mr_factors; // 80  x=metallic, y=roughness, z=normal_scale, w=occlusion_strength
    float4   emissive;   // 96  rgb = emissiveFactor
};
[[vk::push_constant]] Push pc;

[shader("vertex")]
VSOut main(VSIn input)
{
    VSOut output;
    // Slang lays float4x4 row-major while glm is column-major, so apply from the right
    // (mul(v, M) == M * v) — the convention the rest of the engine's shaders use.
    float4 world          = mul(float4(input.position, 1.0), pc.model);
    output.world_position = world.xyz;
    output.position       = mul(world, frame.view_proj);
    // Assumes no non-uniform scale in the model (uniform-scale TRS); a normal matrix would be
    // needed otherwise (noted in findings.md).
    output.world_normal   = normalize(mul(float4(input.normal, 0.0), pc.model).xyz);
    output.world_tangent  = float4(normalize(mul(float4(input.tangent.xyz, 0.0), pc.model).xyz), input.tangent.w);
    output.uv             = input.uv;
    return output;
}
