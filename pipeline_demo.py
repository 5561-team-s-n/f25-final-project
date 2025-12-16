import argparse
from pathlib import Path
import torch
from PIL import Image
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

# MATTE INFERENCE
@torch.no_grad()
def infer_alpha(aim: MatteNet, fg_img: Image.Image, device: str, size: int) -> torch.Tensor:
    fg = utils.pil_to_tensor(fg_img, size=size).to(device)   # (1,3,H,W) in [0,1]
    fg_n = utils.normalize_imagenet(fg, device=device)

    _, _, pred_fusion = aim(fg_n)

    alpha = pred_fusion
    # If output isn't already [0,1] sigmoid it
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
    disable_matteformer: bool = False,
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
    fg = utils.pil_to_tensor(fg_img, size=out_size).to(device) # tensor versions
    bg = utils.pil_to_tensor(bg_img, size=out_size).to(device)

    # GENERATE ALPHA MATTE FOR FOREGROUND
    matte_net = MatteNet()
    ckpt = torch.load(matte_ckpt, map_location="cpu")
    matte_net.load_state_dict(ckpt["state_dict"], strict=True)
    matte_net.to(device).eval()

    alpha_pred = infer_alpha(matte_net, fg_img, device=device, size=out_size)  # (1,1,H,W)

    if debug:
        utils.tensor_gray_to_pil(alpha_pred).save(debug_dir/f"{prefix}_alpha_base.png")
        # np.save(debug_dir/f"{prefix}_alpha_base.npy", alpha_pred.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32))

    #add matteformer
    #--------------

    if matteformer_ckpt and not disable_matteformer:
        if debug: 
            print("Running MatteFormer Refinement...")
        
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
        trimap_np = utils.gen_trimap(alpha_np)
        if debug:
            cv2.imwrite(str(debug_dir/f"{prefix}_trimap_generated.png"), trimap_np)

        # 3. Get FG Image as Numpy (RGB)
        # Resize FG Image to match out_size if it isn't already
        fg_np_pil = fg_img.resize((out_size, out_size), Image.BILINEAR)
        fg_np = np.array(fg_np_pil) # (H,W,3) RGB

        # 4. Prepare Dict
        mf_dict = utils.matteformer_generator_tensor_dict(fg_np, trimap_np)
        
        # 5. Run Inference
        refined_alpha_uint8 = utils.matteformer_inference(mf_model, mf_dict)
        
        if debug:
            cv2.imwrite(str(debug_dir/f"{prefix}_alpha_refined_matteformer.png"), refined_alpha_uint8)

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
        utils.save_depth(depth_fg_full, debug_dir/f"{prefix}_depth_fg_full", debug)
        utils.save_depth(depth_fg,      debug_dir/f"{prefix}_depth_fg_masked", debug)
        utils.save_depth(depth_bg,      debug_dir/f"{prefix}_depth_bg_full", debug)

    # DEPTH-AWARE COMPOSITE OF FOREGROUND ONTO BACKGROUND
    comp = depth_aware_composite(fg, alpha_pred, depth_fg, bg, depth_bg)
    if debug:
        utils.tensor_rgb_to_pil(comp).save(debug_dir/f"{prefix}_comp.png")


    # HARMONIZATION
    mask = alpha_pred.clamp(0.0, 1.0)

    harmonize_net = HarmonizeRunner(ckpt_path=iharm_ckpt, device=device)
    comp_h = harmonize_net.run(comp, mask)

    if debug:
        utils.tensor_rgb_to_pil(comp_h).save(debug_dir / f"{prefix}_iharm.png")

    return utils.tensor_rgb_to_pil(comp_h)

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fg", type=str, default="./samples/fg3.jpg", help="Foreground image path")
    p.add_argument("--bg", type=str, default="./samples/bg3.jpg", help="Background image path")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_size", type=int, default=512)
    p.add_argument("--debug", action="store_true", help="Output the intermediate .npy, depthmap, and alpha mattes for debugging")
    p.add_argument("--disable_matteformer", action="store_true", help="With this option, the pipeline will skip the matteformer alpha matte refinement step.")
    p.add_argument("--disable_cache", action="store_true", help="With this option, the script will avoid reusing depthmaps that have been previously generated. This may considerably slow down things.")
    p.add_argument("--matte_ckpt", type=str, default="./pretrained/matte_net_pretrained.pth")
    p.add_argument("--iharm_ckpt", type=str, default="./pretrained/harmonizer_net_pretrained.pth")
    p.add_argument("--out_dir", type=str, default="./samples/output")
    p.add_argument("--out_filename", type=str, help="Optional output filename (otherwise it'll be like <fg>_<bg>.png)")
    
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
        disable_matteformer=args.disable_matteformer,
        out_dir=args.out_dir,
        disable_cache=args.disable_cache,
    )

    out_dir = Path(args.out_dir)
    if args.out_filename:
        out_path = out_dir / args.out_filename
    else:
        fg_stem = Path(args.fg).stem
        bg_stem = Path(args.bg).stem
        if args.disable_matteformer:
            out_path = out_dir / f"{fg_stem}_{bg_stem}_AIM.png"
        else:
            out_path = out_dir / f"{fg_stem}_{bg_stem}.png"

    out.save(out_path)
    print(f"Saved {out_path}")