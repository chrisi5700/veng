// Fragment stage for veng::passes::PbrPass — Cook-Torrance metallic-roughness BRDF over a single
// directional light plus a flat ambient term (review.md item 5; IBL is a deferred follow-on).
//
// Five textures (set 0, bindings 1..5) feed the standard glTF metallic-roughness inputs; a caller
// with no texture for a slot binds a 1x1 default (white baseColor/MR/occlusion/emissive, flat
// normal) so the shader stays uniform. Each is multiplied by the matching push-constant factor.
// baseColor + emissive are sampled from _SRGB views (decoded to linear by the sampler); normal,
// metallic-roughness and occlusion are _UNORM (linear data) — see review.md item 8.

struct PSIn
{
    float4                     frag_coord     : SV_Position; // framebuffer pixel coord (for the froxel tile)
    [[vk::location(0)]] float3 world_position : POSITION0;
    [[vk::location(1)]] float3 world_normal   : NORMAL0;
    [[vk::location(2)]] float4 world_tangent  : TANGENT0;
    [[vk::location(3)]] float2 uv             : TEXCOORD0;
};

struct Frame
{
    float4x4 view_proj;
    float4x4 view;           // world->view, for the clustered depth slice
    float4   camera_pos;
    float4   light_dir;      // unit direction from surface TO the directional light
    float4   light_color;    // rgb = colour * intensity
    float4   ambient;        // rgb
    uint4    cluster_dims;   // xyz = froxel grid dims, w = cluster count (0 => no clustered point lights)
    float4   cluster_params; // x z_near, y z_far, zw = screen extent (px)
};

// A clustered point light, matching veng::culling::GpuLight (std430: two vec4s).
struct Light
{
    float4 position; // xyz world position, w = radius
    float4 color;    // rgb colour, w = intensity
};

[[vk::binding(0, 0)]] ConstantBuffer<Frame> frame;
[[vk::binding(1, 0)]] Texture2D base_color_tex;
[[vk::binding(2, 0)]] Texture2D normal_tex;
[[vk::binding(3, 0)]] Texture2D mr_tex;        // glTF packing: G = roughness, B = metallic
[[vk::binding(4, 0)]] Texture2D emissive_tex;
[[vk::binding(5, 0)]] Texture2D occlusion_tex; // R = occlusion
[[vk::binding(6, 0)]] SamplerState samp;
// Clustered forward lighting: the per-cluster (offset,count) grid indexes the flat light-index list,
// which indexes the light array. The froxel a fragment belongs to is reconstructed analytically below
// (matching veng::culling), so the froxel AABBs never reach the GPU.
[[vk::binding(7, 0)]] StructuredBuffer<Light> lights;
[[vk::binding(8, 0)]] StructuredBuffer<uint2> light_grid;  // per-cluster (offset, count)
[[vk::binding(9, 0)]] StructuredBuffer<uint>  light_index; // flat light indices

struct Push
{
    float4x4 model;
    float4   base_color;
    float4   mr_factors; // x=metallic, y=roughness, z=normal_scale, w=occlusion_strength
    float4   emissive;   // rgb = emissive colour; w = alpha cutoff (glTF MASK; 0 for OPAQUE/BLEND)
};
[[vk::push_constant]] Push pc;

static const float PI = 3.14159265359;

float3 fresnel_schlick(float cos_theta, float3 f0)
{
    return f0 + (1.0 - f0) * pow(saturate(1.0 - cos_theta), 5.0);
}

float distribution_ggx(float ndoth, float roughness)
{
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = (ndoth * ndoth * (a2 - 1.0) + 1.0);
    return a2 / max(PI * d * d, 1e-7);
}

float geometry_schlick_ggx(float ndotx, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return ndotx / (ndotx * (1.0 - k) + k);
}

float geometry_smith(float ndotv, float ndotl, float roughness)
{
    return geometry_schlick_ggx(ndotv, roughness) * geometry_schlick_ggx(ndotl, roughness);
}

// Cook-Torrance contribution of one light. `l` = unit surface->light direction, `radiance` = incoming
// colour (already attenuated for point lights). Shared by the directional light and every point light.
float3 shade_one(float3 n, float3 v, float3 l, float3 radiance, float3 albedo, float metallic, float roughness,
                 float3 f0)
{
    float3 h     = normalize(v + l);
    float  ndotl = saturate(dot(n, l));
    float  ndotv = saturate(dot(n, v));
    float  ndoth = saturate(dot(n, h));
    float3 fr    = fresnel_schlick(saturate(dot(h, v)), f0);
    float  nd    = distribution_ggx(ndoth, roughness);
    float  ge    = geometry_smith(ndotv, ndotl, roughness);
    float3 specular = (nd * ge * fr) / max(4.0 * ndotv * ndotl, 1e-4);
    float3 kd       = (float3(1.0, 1.0, 1.0) - fr) * (1.0 - metallic);
    float3 diffuse  = kd * albedo / PI;
    return (diffuse + specular) * radiance * ndotl;
}

// The froxel a fragment belongs to — reconstructed analytically to match veng::culling
// (cluster_index + the exponential depth_to_slice), so the cull's per-cluster light lists line up
// with what this shader reads. frag_xy is gl_FragCoord.xy (pixels); world_pos drives the depth slice.
uint froxel_index(float2 frag_xy, float3 world_pos)
{
    uint3  dims   = frame.cluster_dims.xyz;
    float2 extent = frame.cluster_params.zw;
    uint   tx     = min(uint(frag_xy.x / extent.x * float(dims.x)), dims.x - 1);
    uint   ty     = min(uint(frag_xy.y / extent.y * float(dims.y)), dims.y - 1);

    float view_depth = -mul(float4(world_pos, 1.0), frame.view).z;
    float z_near     = frame.cluster_params.x;
    float z_far      = frame.cluster_params.y;
    uint  slice      = 0;
    if (view_depth > z_near)
    {
        float k = log(view_depth / z_near) / log(z_far / z_near) * float(dims.z);
        slice   = min(uint(max(k, 0.0)), dims.z - 1);
    }
    return tx + ty * dims.x + slice * dims.x * dims.y;
}

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    float4 base = base_color_tex.Sample(samp, input.uv) * pc.base_color;
    // glTF alpha MASK: cull fragments below the material's cutoff (pc.emissive.w). OPAQUE and BLEND
    // pass 0 here so this never triggers — BLEND's transparency is the pipeline's alpha blend, and
    // the final base.a (below) is what it blends by.
    if (base.a < pc.emissive.w)
    {
        discard;
    }
    float3 albedo = base.rgb;

    float3 mr        = mr_tex.Sample(samp, input.uv).rgb;
    float  metallic  = saturate(mr.b * pc.mr_factors.x);
    float  roughness = clamp(mr.g * pc.mr_factors.y, 0.04, 1.0);
    float  occlusion = 1.0 + pc.mr_factors.w * (occlusion_tex.Sample(samp, input.uv).r - 1.0);
    float3 emissive  = emissive_tex.Sample(samp, input.uv).rgb * pc.emissive.rgb;

    // Tangent-space normal mapping: rebuild an orthonormal TBN, perturb by the sampled normal.
    float3 n  = normalize(input.world_normal);
    float3 t  = normalize(input.world_tangent.xyz - n * dot(n, input.world_tangent.xyz)); // Gram-Schmidt
    float3 bt = cross(n, t) * input.world_tangent.w;
    float3 tn = normal_tex.Sample(samp, input.uv).xyz * 2.0 - 1.0;
    tn.xy *= pc.mr_factors.z;
    float3 normal = normalize(t * tn.x + bt * tn.y + n * tn.z);

    float3 view = normalize(frame.camera_pos.xyz - input.world_position);
    float3 f0   = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    // Directional key light (always on).
    float3 direct = shade_one(normal, view, normalize(frame.light_dir.xyz), frame.light_color.rgb, albedo, metallic,
                              roughness, f0);

    // Clustered point lights: loop only the lights the cull assigned to this fragment's froxel.
    const uint cluster_count = frame.cluster_dims.w;
    if (cluster_count > 0)
    {
        const uint cluster = froxel_index(input.frag_coord.xy, input.world_position);
        if (cluster < cluster_count)
        {
            const uint2 cell = light_grid[cluster];
            for (uint i = 0; i < cell.y; ++i)
            {
                const Light  lt     = lights[light_index[cell.x + i]];
                const float3 to     = lt.position.xyz - input.world_position;
                const float  dist   = length(to);
                const float  radius = max(lt.position.w, 1e-3);
                const float3 l      = to / max(dist, 1e-4);
                // Smooth windowed inverse-square falloff (UE4-style): fades cleanly to 0 at the radius.
                const float  win = saturate(1.0 - pow(dist / radius, 4.0));
                const float  att = win * win / (dist * dist + 1.0);
                direct += shade_one(normal, view, l, lt.color.rgb * lt.color.w * att, albedo, metallic, roughness, f0);
            }
        }
    }

    float3 ambient = frame.ambient.rgb * albedo * occlusion;
    float3 color   = ambient + direct + emissive;
    return float4(color, base.a);
}
