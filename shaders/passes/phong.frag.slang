// Fragment stage for veng::passes::PhongPass — per-pixel Blinn-Phong, two-sided so the inside
// faces of a translucent object shade correctly. Self-contained lighting (no light UBO): a key
// + fill directional pair over a hemispheric ambient (matching the look of demo/lit_scene.frag),
// plus a Blinn-Phong specular highlight from the key light that tracks the camera — the gloss
// that makes objects read as lit rather than flat/matte.
//
// All inputs are interpolated varyings from phong.vert; nothing is bound here.
struct PSIn
{
    [[vk::location(0)]] float3 world_position : POSITION0;
    [[vk::location(1)]] float3 world_normal   : NORMAL0;
    [[vk::location(2)]] float4 color          : COLOR0;
    [[vk::location(3)]] float4 eye            : POSITION1; // xyz = eye, w = shininess
};

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    float3 n = normalize(input.world_normal);
    float3 v = normalize(input.eye.xyz - input.world_position);
    // Two-sided: a back face (e.g. the inside of a transparent shell) has its normal pointing
    // away from the viewer — flip it so the lighting is computed for the surface we actually see.
    if (dot(n, v) < 0.0)
    {
        n = -n;
    }

    // Two directional lights; each vector points from the surface toward the light.
    float3 key_dir  = normalize(float3(0.4, 0.8, 0.6));
    float3 fill_dir = normalize(float3(-0.5, 0.2, -0.4));
    float  key      = max(dot(n, key_dir), 0.0);
    float  fill     = max(dot(n, fill_dir), 0.0);

    // Hemispheric ambient: ground tone -> sky tone by how upward the normal points.
    float  up      = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    float3 ambient = lerp(float3(0.22, 0.20, 0.18), float3(0.38, 0.40, 0.46), up);

    float3 diffuse = input.color.rgb * (ambient + (key * 0.85) + (fill * 0.35));

    // Blinn-Phong specular from the key light only, gated so back-lit faces get no highlight.
    float  shininess = max(input.eye.w, 1.0);
    float3 half_vec  = normalize(key_dir + v);
    float  spec      = (key > 0.0) ? pow(max(dot(n, half_vec), 0.0), shininess) : 0.0;
    float3 specular  = float3(0.9, 0.9, 0.9) * spec * 0.6;

    return float4(diffuse + specular, input.color.a);
}
