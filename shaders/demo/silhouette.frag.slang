// Silhouette pass: render the cube mesh as a solid white mask (its color input is ignored).
// Paired with demo/mesh.vert (so its VSOut matches), it fills every covered pixel with 1.0 —
// the shape the gaussian blur turns into a glow. The cube's interior cancels out later when
// the ring pass subtracts this same silhouette from the blurred version.
struct PSIn
{
    [[vk::location(0)]] float3 color : COLOR0;
};

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    return float4(1.0, 1.0, 1.0, 1.0);
}
