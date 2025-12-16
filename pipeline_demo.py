import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import toml
import torch
from torch.nn import functional as F

from models.matte_net import MatteNet
from utils.occlusion_composite import depth_aware_composite
from utils.depthpro_runner import DepthProRunner
from utils.harmonizer_runner import HarmonizeRunner


import networks
import utils as utils 
from utils import CONFIG

def pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)


def _normalize_imagenet(x01: torch.Tensor, device: str) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=x01.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - mean) / std


def _tensor_rgb_to_pil(x01: torch.Tensor) -> Image.Image:
    # x01: (1,3,H,W) in [0,1]
    x = x01.detach().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    x = (x * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


def _tensor_gray_to_pil(x01: torch.Tensor) -> Image.Image:
    # x01: (1,1,H,W) in [0,1]
    x = x01.detach().clamp(0.0, 1.0).squeeze(0).squeeze(0).cpu().numpy()
    x = (x * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


def _depth_to_uint8(depth: torch.Tensor, eps: float = 1e-6) -> np.ndarray:
    # depth: (1,1,H,W) float
    d = depth.detach().squeeze(0).squeeze(0).float().cpu().numpy()
    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros_like(d, dtype=np.uint8)
    d_min = float(np.min(d[finite]))
    d_max = float(np.max(d[finite]))
    if d_max <= d_min + eps:
        return np.zeros_like(d, dtype=np.uint8)
    u8 = (255.0 * (d - d_min) / (d_max - d_min + eps)).clip(0, 255).astype(np.uint8)
    return u8

def _annotate_depth_u8_with_legend(u8: np.ndarray, d_min: float, d_max: float, unit: str = "m") -> Image.Image:
    base = Image.fromarray(u8).convert("RGB")
    W, H = base.size
    draw = ImageDraw.Draw(base)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    box_w = int(max(140, min(220, 0.22 * W)))
    box_h = int(max(120, min(260, 0.32 * H)))
    x0, y0 = 10, 10
    x1, y1 = x0 + box_w, y0 + box_h
    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(0, 0, 0), width=1)

    title = f"Depth ({unit})" if unit else "Depth"
    draw.text((x0 + 10, y0 + 6), title, fill=(0, 0, 0), font=font)

    bar_x0 = x0 + 10
    bar_y0 = y0 + 24
    bar_w = 16
    bar_y1 = y1 - 10
    bar_h = max(1, bar_y1 - bar_y0)

    for j in range(bar_h):
        t = j / (bar_h - 1) if bar_h > 1 else 0.0
        shade = int(round(255.0 * (1.0 - t)))
        draw.line([(bar_x0, bar_y0 + j), (bar_x0 + bar_w, bar_y0 + j)], fill=(shade, shade, shade))
    draw.rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h], outline=(0, 0, 0), width=1)

    ticks = [1.0, 0.75, 0.5, 0.25, 0.0]  # 1=max at top, 0=min at bottom
    for p in ticks:
        y = int(round(bar_y0 + (1.0 - p) * (bar_h - 1)))
        draw.line([(bar_x0 + bar_w + 2, y), (bar_x0 + bar_w + 8, y)], fill=(0, 0, 0))
        val = d_min + p * (d_max - d_min)
        label = f"{val:.3g}" + (f" {unit}" if unit else "")
        draw.text((bar_x0 + bar_w + 12, y - 6), label, fill=(0, 0, 0), font=font)

    return base


def _save_depth(depth: torch.Tensor, out_base: Path, debug: bool):
    """
    Writes:
      out_base.with_suffix(".npy")  (float32 HxW)
      out_base.with_suffix(".png")  (min-max normalized grayscale)
    """
    out_base.parent.mkdir(parents=True, exist_ok=True)

    d = depth.detach().squeeze(0).squeeze(0).float().cpu().numpy().astype(np.float32)
    np.save(out_base.with_suffix(".npy"), d)

    finite = np.isfinite(d)
    if np.any(finite):
        d_min = float(np.min(d[finite]))
        d_max = float(np.max(d[finite]))
    else:
        d_min, d_max = 0.0, 0.0
    
    if debug:
        print(f"Min/max depth for {out_base}: {d_min}, {d_max}")

    u8 = _depth_to_uint8(depth)
    _annotate_depth_u8_with_legend(u8, d_min=d_min, d_max=d_max, unit="m").save(out_base.with_suffix(".png"))

#MATTEFORMER helpers

def gen_trimap(alpha, max_kernel_size=30):
    """
    Generate a trimap using MatteFormer's erosion method.
    alpha: uint8 grayscale alpha matte (0–255)
    """
    alpha = alpha.astype(np.float32) / 255.0

    # Foreground mask = alpha = 1
    fg_mask = (alpha + 1e-5).astype(int).astype(np.uint8)
    # Background mask = alpha = 0
    bg_mask = (1 - alpha + 1e-5).astype(int).astype(np.uint8)

    # Prebuild kernels like MatteFormer
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        for ks in range(1, max_kernel_size + 1)
    ]

    # Randomly choose kernel sizes (or fix them if deterministic behavior is preferred)
    # Using random here as per your original script
    fg_erode_kernel = kernels[np.random.randint(1, max_kernel_size)]
    bg_erode_kernel = kernels[np.random.randint(1, max_kernel_size)]

    fg_mask_eroded = cv2.erode(fg_mask, fg_erode_kernel)
    bg_mask_eroded = cv2.erode(bg_mask, bg_erode_kernel)

    # Create the trimap: 128 = unknown
    trimap = np.ones_like(alpha, dtype=np.uint8) * 128
    trimap[fg_mask_eroded == 1] = 255
    trimap[bg_mask_eroded == 1] = 0

    return trimap

def matteformer_generator_tensor_dict(image, trimap):
    """
    Prepare dictionary for MatteFormer using in-memory Numpy arrays.
    """
    sample = {'image': image, 'trimap': trimap, 'alpha_shape': (image.shape[0], image.shape[1])}
    h, w = sample["alpha_shape"]
    
    # Calculate Padding to multiple of 32
    if h % 32 == 0 and w % 32 == 0:
        padded_image = np.pad(sample['image'], ((32,32), (32, 32), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32,32), (32, 32)), mode="reflect")
    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        padded_image = np.pad(sample['image'], ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
        padded_trimap = np.pad(sample['trimap'], ((32,pad_h+32), (32, pad_w+32)), mode="reflect")

    sample['image'] = padded_image
    sample['trimap'] = padded_trimap

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    # Convert RGB image (PIL loaded RGB) to standard float32

    
    image = sample['image'].transpose((2, 0, 1)).astype(np.float32)
    trimap = sample['trimap']

    # Trimap configuration
    trimap[trimap < 85] = 0
    trimap[trimap >= 170] = 2
    trimap[trimap >= 85] = 1

    image /= 255.

    # To tensor
    img_t = torch.from_numpy(image)
    tri_t = torch.from_numpy(trimap).to(torch.long)
    
    img_t = img_t.sub_(mean).div_(std)
    
    # Trimap to one-hot 3 channel
    tri_t = F.one_hot(tri_t, num_classes=3).permute(2, 0, 1).float()

    # Add Batch Dimension
    sample['image'] = img_t.unsqueeze(0)
    sample['trimap'] = tri_t.unsqueeze(0)

    return sample

def matteformer_inference(model, image_dict):
    with torch.no_grad(): 
        image, trimap = image_dict['image'], image_dict['trimap']
        image = image.cuda()
        trimap = trimap.cuda()

        pred = model(image, trimap)
        alpha_pred_os1 = pred['alpha_os1']
        alpha_pred_os4 = pred['alpha_os4']
        alpha_pred_os8 = pred['alpha_os8']

        # Refinement
        alpha_pred = alpha_pred_os8.clone().detach()
        weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width1, train_mode=False)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        
        weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width2, train_mode=False)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]

        h, w = image_dict['alpha_shape']
        alpha_pred = alpha_pred[0, 0, ...].data.cpu().numpy() * 255
        alpha_pred = alpha_pred.astype(np.uint8)

        # Enforce trimap constraints
        # 0: bg, 2: fg
        alpha_pred[np.argmax(trimap.cpu().numpy()[0], axis=0) == 0] = 0.0
        alpha_pred[np.argmax(trimap.cpu().numpy()[0], axis=0) == 2] = 255.

        # Crop padding
        alpha_pred = alpha_pred[32:h+32, 32:w+32]

        return alpha_pred

# MATTE INFERENCE
@torch.no_grad()
def _aim_infer_alpha(aim: MatteNet, fg_img: Image.Image, device: str, size: int) -> torch.Tensor:
    fg = pil_to_tensor(fg_img, size=size).to(device)   # (1,3,H,W) in [0,1]
    fg_n = _normalize_imagenet(fg, device=device)

    pred_global, pred_local, pred_fusion = aim(fg_n)

    alpha = pred_fusion
    # If output isn't already [0,1], sigmoid it.
    if float(alpha.min().item()) < 0.0 or float(alpha.max().item()) > 1.0:
        alpha = torch.sigmoid(alpha)
    return alpha.clamp(0.0, 1.0)

# MAIN PIPELINE
@torch.no_grad()
def run_pipeline(
    fg_path,
    bg_path,
    device="cuda",
    out_size=512,
    matte_ckpt="./pretrained/matte_net_pretrained.pth",
    iharm_ckpt="./pretrained/harmonizer_net_pretrained.pth",
    debug: bool = False,
    matteformer_disable: bool = False,
    matteformer_ckpt="./pretrained/matteformer_pretrained_1k.pth",
    out_dir: str = "./samples/output",
    disable_cache: bool = False,
):
    # SETUP DIRECTORIES AND FILENAMES
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    if not disable_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        
    fg_path = Path(fg_path)
    bg_path = Path(bg_path)
    prefix = f"{fg_path.stem}__{bg_path.stem}"


    # SET UP PARAMETERS
    # IMAGE OUTPUT SIZE
    # (size of our image has to be divisible by 32 to work with the model)
    if out_size % 32 != 0:
        out_size = max(32, out_size - (out_size % 32))
    # TORCH DEVICE
    device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    if debug:
        print(f"Using device: {device}")


    # LOAD IMAGES
    fg_img = Image.open(fg_path).convert("RGB")
    bg_img = Image.open(bg_path).convert("RGB")
    fg = pil_to_tensor(fg_img, size=out_size).to(device) # tensor versions
    bg = pil_to_tensor(bg_img, size=out_size).to(device)

    if debug:
        # better to output the round-tripped version rather than the original fg_img and bg_img
        # because this also lets us spot issues with our tensor conversion code 
        _tensor_rgb_to_pil(fg).save(debug_dir/f"{prefix}_fg_resized.png")
        _tensor_rgb_to_pil(bg).save(debug_dir/f"{prefix}_bg_resized.png")


    # GENERATE ALPHA MATTE FOR FOREGROUND
    matte_net = MatteNet()
    ckpt = torch.load(matte_ckpt, map_location="cpu")
    matte_net.load_state_dict(ckpt["state_dict"], strict=True)
    matte_net.to(device).eval()

    alpha_pred = _aim_infer_alpha(matte_net, fg_img, device=device, size=out_size)  # (1,1,H,W)

    if debug:
        _tensor_gray_to_pil(alpha_pred).save(debug_dir/f"{prefix}_alpha_aim.png")
        np.save(debug_dir/f"{prefix}_alpha_aim.npy", alpha_pred.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32))

    #add matteformer
    #--------------

    if matteformer_ckpt and not matteformer_disable:
        if debug: print("Running MatteFormer Refinement...")
        
        # Load Config
        with open("./config/MatteFormer_Composition1k.toml", encoding="utf-8") as f:
            utils.load_config(toml.load(f))
        
        # Build Model
        mf_model = networks.get_generator(is_train=False)
        mf_model.cuda()
        mf_checkpoint = torch.load(matteformer_ckpt)
        mf_model.load_state_dict(utils.remove_prefix_state_dict(mf_checkpoint['state_dict']), strict=True)
        mf_model = mf_model.eval()

        # Prep Data for MatteFormer
        # 1. Convert Tensor Alpha to Numpy UInt8 for Trimap Gen
        alpha_np = (alpha_pred * 255).byte().cpu().numpy()[0, 0] # (H,W)
        
        # 2. Generate Trimap
        trimap_np = gen_trimap(alpha_np)
        if debug:
            cv2.imwrite(str(debug_dir/f"{prefix}_trimap_generated.png"), trimap_np)

        # 3. Get FG Image as Numpy (RGB)
        # Resize FG Image to match out_size if it isn't already
        fg_np_pil = fg_img.resize((out_size, out_size), Image.BILINEAR)
        fg_np = np.array(fg_np_pil) # (H,W,3) RGB

        # 4. Prepare Dict
        mf_dict = matteformer_generator_tensor_dict(fg_np, trimap_np)
        
        # 5. Run Inference
        refined_alpha_uint8 = matteformer_inference(mf_model, mf_dict)
        
        if debug:
            cv2.imwrite(str(debug_dir/f"{prefix}_alpha_02_matteformer_refined.png"), refined_alpha_uint8)

        # 6. Convert back to Tensor (1,1,H,W) [0-1] for pipeline
        alpha_pred = torch.from_numpy(refined_alpha_uint8).float().to(device) / 255.0
        alpha_pred = alpha_pred.unsqueeze(0).unsqueeze(0)



    # GENERATE DEPTHMAP FOR FOREGROUND AND BACKGROUND
    depth_runner = DepthProRunner(device=device)
    target_hw = (fg.shape[-2], fg.shape[-1])

    fg_cache = cache_dir / f"{fg_path.stem}_depth.npy"
    depth_fg_full = None
    # Try to fetch from foreground depth cache:
    if not disable_cache and fg_cache.exists():
        arr = np.load(fg_cache)
        if arr.ndim == 2 and tuple(arr.shape) == target_hw:
            depth_fg_full = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            if debug:
                print(f"Loaded cached FG depth: {fg_cache}")

    # No cached foreground depth cache was found, or we want to ignore cache
    if depth_fg_full is None:
        depth_fg_full = depth_runner.run(fg_img, target_size=target_hw).to(device)  # (1,1,H,W)
        if not disable_cache:
            np.save(fg_cache, depth_fg_full.detach().squeeze(0).squeeze(0).float().cpu().numpy().astype(np.float32))
            if debug:
                print(f"Saved FG depth cache: {fg_cache}")
    # Apply alpha matte to depthmap (!!)
    depth_fg = depth_fg_full * alpha_pred

    bg_cache = cache_dir / f"{bg_path.stem}_depth.npy"
    depth_bg = None
    # Try to fetch from background depth cache:
    if not disable_cache and bg_cache.exists():
        arr = np.load(bg_cache)
        if arr.ndim == 2 and tuple(arr.shape) == target_hw:
            depth_bg = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            if debug:
                print(f"Loaded cached BG depth: {bg_cache}")

    #  No cached background depth cache was found, or we want to ignore cache
    if depth_bg is None:
        depth_bg = depth_runner.run(bg_img, target_size=target_hw).to(device)       # (1,1,H,W)
        if not disable_cache:
            np.save(bg_cache, depth_bg.detach().squeeze(0).squeeze(0).float().cpu().numpy().astype(np.float32))
            if debug:
                print(f"Saved BG depth cache: {bg_cache}")

    if debug:
        _save_depth(depth_fg_full, debug_dir/f"{prefix}_depth_fg_full", debug)
        _save_depth(depth_fg,      debug_dir/f"{prefix}_depth_fg_masked", debug)
        _save_depth(depth_bg,      debug_dir/f"{prefix}_depth_bg_full", debug)

    # DEPTH-AWARE COMPOSITE OF FOREGROUND ONTO BACKGROUND
    comp = depth_aware_composite(fg, alpha_pred, depth_fg, bg, depth_bg)
    if debug:
        _tensor_rgb_to_pil(comp).save(debug_dir/f"{prefix}_composite_depthaware.png")


    # HARMONIZATION
    mask = alpha_pred.clamp(0.0, 1.0)

    aict = HarmonizeRunner(ckpt_path=iharm_ckpt, device=device)
    comp_h = aict.run(comp, mask)

    if debug:
        _tensor_rgb_to_pil(comp_h).save(debug_dir / f"{prefix}_harmonized_aict.png")

    return _tensor_rgb_to_pil(comp_h)

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fg", type=str, default="./samples/foreground3.jpg", help="Foreground image path")
    p.add_argument("--bg", type=str, default="./samples/background3.jpg", help="Background image path")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_size", type=int, default=512)
    p.add_argument("--debug", action="store_true", help="Output the intermediate .npy, depthmap, and alpha mattes for debugging")
    p.add_argument("--matteformer_disable", action="store_true", help="With this option, the pipeline will skip the mattformer alphamat refinement step.")
    p.add_argument("--disable_cache", action="store_true", help="With this option, the script will avoid reusing depthmaps that have been previously generated. This may considerably slow down things.")
    p.add_argument("--matte_ckpt", type=str, default="./pretrained/matte_net_pretrained.pth")
    p.add_argument("--iharm_ckpt", type=str, default="./pretrained/harmonizer_net_pretrained.pth")
    p.add_argument("--out_dir", type=str, default="./samples/output")
    p.add_argument("--out_filename", type=str, help="Optional output filename (otherwise it'll be like <fg>__<bg>_result.png)")
    
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    out = run_pipeline(
        fg_path=args.fg,
        bg_path=args.bg,
        device=args.device,
        out_size=args.out_size,
        matte_ckpt=args.matte_ckpt,
        iharm_ckpt=args.iharm_ckpt,
        debug=args.debug,
        matteformer_disable=args.matteformer_disable,
        out_dir=args.out_dir,
        disable_cache=args.disable_cache,
    )

    out_dir = Path(args.out_dir)
    if args.out_filename:
        out_path = out_dir / args.out_filename
    else:
        fg_stem = Path(args.fg).stem
        bg_stem = Path(args.bg).stem
        if args.matteformer_disable:
            out_path = out_dir / f"{fg_stem}__{bg_stem}_resultAIM.png"
        else:
            out_path = out_dir / f"{fg_stem}__{bg_stem}_resultMATTF.png"

    out.save(out_path)
    print(f"Saved {out_path}")