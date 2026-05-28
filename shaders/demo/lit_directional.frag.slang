// Fragment shader pairing with lit_instanced.vert: one directional light + a small ambient
// term. The light direction comes from a uniform buffer (descriptor set 0, binding 0)
// uploaded by a UniformNode — keeping it on a reactive edge means changing the sun
// direction at runtime invalidates the draw exactly the same way `tint` does in the
// existing demo. The per-vertex color (the cube's face color) is what's lit; with white
// bodies the output is the shade.
struct PSIn
{
    [[vk::location(0)]] float3 world_position : POSITION0;
    [[vk::location(1)]] float3 world_normal   : NORMAL0;
    [[vk::location(2)]] float3 color          : COLOR0;
};

struct Light
{
    // direction is the *direction the light travels in* (the "sun-to-surface" vector); we
    // dot the inverse (surface-to-sun) with the world-space normal for Lambert. .a unused.
    float4 direction;
};
[[vk::binding(0, 0)]] ConstantBuffer<Light> light;

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    float3 N        = normalize(input.world_normal);
    float3 to_light = normalize(-light.direction.xyz);
    float  lambert  = max(0.0, dot(N, to_light));

    // 0.15 ambient floor so back faces are still visible (otherwise dark cubes vanish
    // into the clear color and the scene looks empty).
    float  shade = 0.15 + 0.85 * lambert;
    return float4(input.color * shade, 1.0);
}
