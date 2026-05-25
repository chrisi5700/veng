// Slice triangle: a centered triangle generated from the vertex index (no vertex
// buffer). Covers the middle of the viewport, leaving the corners to the clear color.
struct VSOut
{
    float4 position : SV_Position;
};

[shader("vertex")]
VSOut main(uint vid : SV_VertexID)
{
    float2 positions[3] = {
        float2( 0.0, -0.6),
        float2( 0.6,  0.6),
        float2(-0.6,  0.6),
    };
    VSOut output;
    output.position = float4(positions[vid], 0.0, 1.0);
    return output;
}
