import numpy as np
from pathlib import Path

import OpenEXR, Imath
from PIL import Image

# ---- paths (edit if needed) ----
BG_EXR     = "Balls.exr"
FG_RGB_PNG = "fg_rgb.jpg"
FG_ALPHA_PNG = "fg_alpha.png"
FG_DEPTH_NPY = "fg_depth.npy"
OUT_RGB_PNG = "composite_balls_fg.png"
OUT_VIS_BG_Z_PNG = "balls_depth_vis.png"
OUT_VIS_FG_Z_PNG = "fg_depth_vis.png"
# --------------------------------


def load_exr_rgbz(path):
    exr = OpenEXR.InputFile(path)
    hdr = exr.header()
    dw = hdr["dataWindow"]
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    def ch(name):
        return np.frombuffer(exr.channel(name, FLOAT), dtype=np.float32).reshape(h, w)

    R = ch("R")
    G = ch("G")
    B = ch("B")
    Z = ch("Z")  # depth
    rgb = np.stack([R, G, B], axis=-1)  # H,W,3
    return rgb, Z


def resize_np_image(arr, size_hw):
    """Resize HxWxC or HxW with PIL."""
    h, w = size_hw
    if arr.ndim == 3:
        img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype("uint8"))
        img = img.resize((w, h), Image.BILINEAR)
        out = np.array(img).astype("float32") / 255.0
    else:
        img = Image.fromarray(arr.astype("float32"))
        img = img.resize((w, h), Image.BILINEAR)
        out = np.array(img).astype("float32")
    return out


def minmax_vis(d):
    d_min, d_max = float(d.min()), float(d.max())
    if d_max <= d_min + 1e-6:
        return np.zeros_like(d, dtype="uint8")
    return ((d - d_min) / (d_max - d_min + 1e-6) * 255.0).clip(0, 255).astype("uint8")


def main():
    # 1) load background RGB + depth
    bg_rgb, bg_z = load_exr_rgbz(BG_EXR)  # bg_rgb: H,W,3 in float32, bg_z: H,W
    H, W, _ = bg_rgb.shape

    # 2) load foreground RGB, alpha, depth
    fg_rgb = np.array(Image.open(FG_RGB_PNG).convert("RGB")).astype("float32") / 255.0  # H',W',3
    fg_alpha = np.array(Image.open(FG_ALPHA_PNG).convert("L")).astype("float32")
    if fg_alpha.max() > 1.0:
        fg_alpha = fg_alpha / 255.0
    fg_depth = np.load(FG_DEPTH_NPY).astype("float32")  # H'',W''

    # 3) resize foreground stuff to background size
    fg_rgb_resized = resize_np_image(fg_rgb, (H, W))  # H,W,3
    fg_alpha_resized = resize_np_image(fg_alpha, (H, W))  # H,W
    fg_depth_resized = resize_np_image(fg_depth, (H, W))  # H,W

    # 4) simple depth alignment (optional, tweak if needed)
    # here we just assume: smaller depth = closer; we can bias fg "closer"
    depth_bias = 0.0  # make negative to force FG in front
    fg_z = fg_depth_resized + depth_bias

    # 5) depth-aware alpha composite
    # bg is fully opaque (alpha=1), so:
    # if FG is in front (fg_z < bg_z), composite FG over BG
    front_fg = (fg_z < bg_z) & (fg_alpha_resized > 1e-3)

    # broadcast alpha / masks
    alpha = fg_alpha_resized[..., None]  # H,W,1
    front_fg_3 = front_fg[..., None].astype("float32")

    # standard "over": out = fg*alpha + bg*(1-alpha), but only where FG is in front
    comp_rgb = bg_rgb.copy()
    fg_contrib = fg_rgb_resized * alpha + bg_rgb * (1.0 - alpha)
    comp_rgb = front_fg_3 * fg_contrib + (1.0 - front_fg_3) * bg_rgb

    # 6) save outputs
    Image.fromarray(np.clip(comp_rgb * 255.0, 0, 255).astype("uint8")).save(OUT_RGB_PNG)
    Image.fromarray(minmax_vis(bg_z)).save(OUT_VIS_BG_Z_PNG)
    Image.fromarray(minmax_vis(fg_depth_resized)).save(OUT_VIS_FG_Z_PNG)

    print("Saved composite:", OUT_RGB_PNG)
    print("Saved BG depth vis:", OUT_VIS_BG_Z_PNG)
    print("Saved FG depth vis:", OUT_VIS_FG_Z_PNG)


if __name__ == "__main__":
    main()
