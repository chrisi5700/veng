// Spinning-cube vertex shader for the reactive renderer demo. No vertex buffer: the
// 36 vertices (12 triangles, 6 faces) are generated from SV_VertexID. The combined
// model-view-projection matrix arrives as a push constant — a glm::mat4 (column-major)
// uploaded raw; Slang lays the push constant out row-major, so the shader applies it
// from the right (mul(vector, matrix)) to recover M * v (see below).
struct VSOut
{
    float4 position : SV_Position;
    [[vk::location(0)]] float3 color : COLOR0;
};

struct PushData
{
    float4x4 mvp;
};
[[vk::push_constant]] PushData pc;

[shader("vertex")]
VSOut main(uint vid : SV_VertexID)
{
    const float3 corners[8] = {
        float3(-0.5, -0.5, -0.5), float3( 0.5, -0.5, -0.5),
        float3( 0.5,  0.5, -0.5), float3(-0.5,  0.5, -0.5),
        float3(-0.5, -0.5,  0.5), float3( 0.5, -0.5,  0.5),
        float3( 0.5,  0.5,  0.5), float3(-0.5,  0.5,  0.5),
    };
    const uint indices[36] = {
        0, 1, 2, 0, 2, 3, // front  (-z)
        5, 4, 7, 5, 7, 6, // back   (+z)
        4, 0, 3, 4, 3, 7, // left   (-x)
        1, 5, 6, 1, 6, 2, // right  (+x)
        4, 5, 1, 4, 1, 0, // bottom (-y)
        3, 2, 6, 3, 6, 7, // top    (+y)
    };
    const float3 faceColors[6] = {
        float3(1.0, 0.25, 0.25), float3(0.25, 1.0, 0.35),
        float3(0.30, 0.45, 1.0), float3(1.0, 0.85, 0.25),
        float3(1.0, 0.35, 1.0),  float3(0.30, 1.0, 1.0),
    };

    VSOut output;
    // Slang lays out the push-constant float4x4 row-major, so the column-major glm
    // matrix arrives transposed; applying it from the right (mul(v, M)) computes M * v.
    output.position = mul(float4(corners[indices[vid]], 1.0), pc.mvp);
    output.color = faceColors[vid / 6];
    return output;
}
