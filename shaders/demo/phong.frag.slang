// Per-fragment Phong shading. Material is shared via index — the vert forwards the per-
// instance material_index (with nointerpolation, since it's an integer), and we look up
// materials[material_index] here. That index is set per OBJECT on the CPU side, so a
// shared "gold" or "plastic" material is one entry referenced by many objects.
//
// Bindings:
//   * materials : SSBO, one entry per *material* (not per object) — shared by all draws.
//   * light     : ConstantBuffer (UniformNode), scene-wide light state.
//   * view      : ConstantBuffer (UniformNode), driven by the orbit camera's eye_pos so
//                 specular highlights track the viewer.
struct PSIn
{
    [[vk::location(0)]] nointerpolation uint material_index : COLOR0;
    [[vk::location(1)]]                 float3 world_position : POSITION0;
    [[vk::location(2)]]                 float3 world_normal   : NORMAL0;
};

struct Material
{
    float4 ambient;  // .xyz = color, .w = Ka
    float4 diffuse;  // .xyz = color, .w = Kd
    float4 specular; // .xyz = color, .w = Ks
    float4 params;   // .x = shininess, .y = mode (1 = full)
};
[[vk::binding(3, 0)]] StructuredBuffer<Material> materials;

struct Light
{
    float4 position; // .xyz = world position
    float4 color;    // .xyz = color (intensity folded in)
};
[[vk::binding(4, 0)]] ConstantBuffer<Light> light;

struct View
{
    float4 eye; // .xyz = world-space camera position
};
[[vk::binding(5, 0)]] ConstantBuffer<View> view;

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    Material m = materials[input.material_index];

    float3 N = normalize(input.world_normal);
    float3 L = normalize(light.position.xyz - input.world_position);

    float lambert  = max(dot(N, L), 0.0);
    float specular = 0.0;
    if (lambert > 0.0)
    {
        float3 R        = reflect(-L, N);
        float3 V        = normalize(view.eye.xyz - input.world_position);
        float  spec_dot = max(dot(R, V), 0.0);
        specular        = pow(spec_dot, m.params.x);
    }

    float Ka = m.ambient.w;
    float Kd = m.diffuse.w;
    float Ks = m.specular.w;

    float3 col = Ka * m.ambient.xyz +
                 Kd * lambert * m.diffuse.xyz * light.color.xyz +
                 Ks * specular * m.specular.xyz * light.color.xyz;

    return float4(col, 1.0);
}
