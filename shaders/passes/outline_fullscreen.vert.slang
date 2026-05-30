// Fullscreen-triangle vertex shader for the outline post-process passes: no vertex buffer,
// three vertices fabricated from SV_VertexID cover the whole viewport. Emits a [0,1] UV with
// (0,0) at the top-left — the same orientation the offscreen passes render in (Vulkan y-down
// clip), so no flip is needed when sampling them. Used by outline_blur + outline_ring with
// vertex_count = 3.
struct VSOut
{
    float4              position : SV_Position;
    [[vk::location(0)]] float2 uv : TEXCOORD0;
};

[shader("vertex")]
VSOut main(uint vid : SV_VertexID)
{
    // vid 0,1,2 -> uv (0,0),(2,0),(0,2) -> pos (-1,-1),(3,-1),(-1,3): a triangle covering the
    // [-1,1] square; the [0,1] UV sub-range maps exactly onto the screen.
    const float2 uv = float2((vid << 1) & 2, vid & 2);
    VSOut        output;
    output.uv       = uv;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}
