#!/usr/bin/env python3
"""Pack a glTF-style metallic-roughness texture from ambientCG's separate maps.

veng's PBR shader (shaders/passes/pbr.frag.slang) samples one `metal_rough` texture with the glTF
channel packing: G = roughness, B = metallic. ambientCG ships these as two separate greyscale PNGs,
so this combines them into a single linear RGBA8 image (R=0, G=roughness, B=metalness, A=255).

Usage: pack_mr.py <roughness.png> <metalness.png> <out_mr.png>
"""

import sys
from PIL import Image


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    rough_path, metal_path, out_path = sys.argv[1:4]

    rough = Image.open(rough_path).convert("L")
    metal = Image.open(metal_path).convert("L")
    if metal.size != rough.size:
        metal = metal.resize(rough.size, Image.BILINEAR)

    width, height = rough.size
    zero = Image.new("L", (width, height), 0)
    full = Image.new("L", (width, height), 255)
    # R unused, G = roughness, B = metalness, A = opaque.
    Image.merge("RGBA", (zero, rough, metal, full)).save(out_path)
    print(f"packed {out_path} ({width}x{height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
