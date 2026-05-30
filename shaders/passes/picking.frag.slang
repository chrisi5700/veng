// Picking pass fragment shader for veng::passes::PickingPass: encode the per-object id
// (forwarded flat from the vertex shader) into the RGBA8 color attachment, little-endian byte
// order, so a CPU readback of those 4 bytes reconstructs the id exactly.
//
// The attachment MUST be a UNORM (linear, non-sRGB) format so that float(byte)/255 round-trips
// back to the exact byte. Background pixels keep the pass's clear color (0,0,0,0), which decodes
// to id 0 — the reserved "no hit" value (object ids must be >= 1).
struct PSIn
{
    [[vk::location(0)]] nointerpolation uint id : COLOR0;
};

[shader("fragment")]
float4 main(PSIn input) : SV_Target
{
    const uint id = input.id;
    return float4(float(id & 0xFFu) / 255.0,         // byte 0 -> R
                  float((id >> 8) & 0xFFu) / 255.0,  // byte 1 -> G
                  float((id >> 16) & 0xFFu) / 255.0, // byte 2 -> B
                  float((id >> 24) & 0xFFu) / 255.0);// byte 3 -> A
}
