import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

# use this file like:
# python generate_pseudo_depths.py --save_png --image_root ../datasets/Distinctions-646/Generated/Test/Image/ --out_root ../datasets/Distinctions-646/Generated/Test/DepthPro/ 
# python generate_pseudo_depths.py --image_root ../datasets/AIM-500/original/ --mask_root  ../datasets/AIM-500/mask/  --out_root ../datasets/AIM-500/depthpro/ --device cuda --save_png

def find_mask_for_image(mask_root: Path, rel_img: Path):
    """
    rel_img: e.g. foo/bar.jpg under original/
    We look under mask_root/foo/ for any file with stem 'bar.*'
    so masks can be .png while originals are .jpg, etc.
    """
    mask_dir = mask_root / rel_img.parent
    stem = rel_img.stem
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    candidates = list(mask_dir.glob(stem + ".*"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No mask found for image {rel_img} under {mask_dir}")
    if len(candidates) > 1:
        print(f"[warn] Multiple masks for {rel_img}, using {candidates[0]}")
    return candidates[0]


def save_depth_png(depth_arr: np.ndarray, out_path: Path):
    """
    depth_arr: (H,W) float32, arbitrary scale.
    Saves a min-max normalized grayscale PNG to out_path (with .png suffix).
    """
    d = depth_arr
    d_min, d_max = float(d.min()), float(d.max())
    if d_max <= d_min + 1e-6:
        # degenerate case: flat map
        d_vis = np.zeros_like(d, dtype="uint8")
    else:
        d_vis = (255.0 * (d - d_min) / (d_max - d_min + 1e-6)).clip(0, 255).astype("uint8")

    d_png = Image.fromarray(d_vis)
    out_path = out_path.with_suffix(".png")
    d_png.save(out_path)


def process_folder(image_root, mask_root, out_root, device="cuda", save_png=False):
    image_root = Path(image_root)
    mask_root = Path(mask_root)
    out_root = Path(out_root)

    if not image_root.is_dir():
        raise RuntimeError(f"image_root is not a directory or does not exist: {image_root}")
    if not mask_root.is_dir():
        raise RuntimeError(f"mask_root is not a directory or does not exist: {mask_root}")

    image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    model = DepthProForDepthEstimation.from_pretrained(
        "apple/DepthPro-hf", use_fov_model=False
    ).to(device)
    model.eval()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # collect all image files under original/
    image_paths = [
        p for p in image_root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]

    print(f"Found {len(image_paths)} images under {image_root}")

    with torch.no_grad():
        for img_path in tqdm(image_paths):
            # relative path inside original/
            rel = img_path.relative_to(image_root)

            # output path base under out_root, mirroring directory structure
            out_base = out_root / rel
            out_base = out_base.with_suffix("")  # strip .png/.jpg/etc
            out_base.parent.mkdir(parents=True, exist_ok=True)

            full_npy = out_base.parent / (out_base.name + "_depth_full.npy")
            fg_npy = out_base.parent / (out_base.name + "_depth_fg.npy")

            # skip if both npy outputs already exist
            if full_npy.exists() and fg_npy.exists():
                continue

            # load RGB image
            image = Image.open(img_path).convert("RGB")

            # find and load corresponding matte
            mask_path = find_mask_for_image(mask_root, rel)
            mask_img = Image.open(mask_path).convert("L")
            # ensure same size as original image
            if mask_img.size != image.size:
                mask_img = mask_img.resize(image.size, Image.NEAREST)

            alpha = np.array(mask_img).astype("float32") / 255.0  # H,W in [0,1]

            # prepare inputs for DepthPro
            inputs = image_processor(images=image, return_tensors="pt").to(device)

            # run Depth Pro on the full image
            outputs = model(**inputs)

            # post-process to original size
            post = image_processor.post_process_depth_estimation(
                outputs, target_sizes=[(image.height, image.width)]
            )
            depth = post[0]["predicted_depth"]  # (H, W), float32 tensor

            depth = depth.cpu().numpy().astype("float32")  # metric-like depth, H,W

            # full-scene depth (raw)
            depth_full = depth

            # foreground-only depth: apply matte
            depth_fg = depth_full * alpha  # background becomes ~0

            # optional PNG visualizations
            if save_png:
                full_png = out_base.parent / (out_base.name + "_depth_full")
                fg_png = out_base.parent / (out_base.name + "_depth_fg")
                save_depth_png(depth_full, full_png)
                save_depth_png(depth_fg, fg_png)

            # save npy
            np.save(full_npy, depth_full)
            np.save(fg_npy, depth_fg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Path to AIM-500 original images (e.g., src/datasets/AIM-500/original)",
    )
    parser.add_argument(
        "--mask_root",
        type=str,
        required=True,
        help="Path to AIM-500 ground-truth mattes (e.g., src/datasets/AIM-500/mask)",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Where to mirror depth maps (e.g., src/datasets/AIM-500/DepthPro)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save_png",
        action="store_true",
        help="Also save min-max normalized grayscale PNG visualizations for full and fg depth.",
    )
    args = parser.parse_args()

    device = args.device
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        device = "cpu"

    process_folder(
        args.image_root,
        args.mask_root,
        args.out_root,
        device=device,
        save_png=args.save_png,
    )