import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from models.mattedepth_net_poc import MatDepthNet

# use like:
# python infer_mat_depth.py --ckpt ./pretrained/matte_depth_pretrained.pt --image ./datasets/AIM-500/original/o_a3132475.jpg --out_dir ./outputs_infer --device cuda

@torch.no_grad()
def run_inference(
    ckpt_path: str,
    image_path: str,
    out_dir: str,
    image_size: int = 512,
    device: str = "cuda",
):
    device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"

    # load model
    model = MatDepthNet(pretrained_encoder=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # load image
    img_p = Path(image_path)
    img = Image.open(img_p).convert("RGB")
    orig_w, orig_h = img.size

    img_resized = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(img_resized).astype("float32") / 255.0     # H,W,3 in [0,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # 1,3,H,W

    # 3. Forward pass
    alpha_pred, depth_pred = model(tensor)  # 1,1,H,W each

    # 4. Resize predictions back to original size
    alpha_pred = torch.nn.functional.interpolate(alpha_pred, size=(orig_h, orig_w),
                               mode="bilinear", align_corners=False)
    depth_pred = torch.nn.functional.interpolate(depth_pred, size=(orig_h, orig_w),
                               mode="bilinear", align_corners=False)

    alpha_np = alpha_pred.squeeze(0).squeeze(0).cpu().numpy()  # H,W
    depth_np = depth_pred.squeeze(0).squeeze(0).cpu().numpy()  # H,W

    # 5. Convert to displayable PNGs
    # Alpha in [0,1] -> uint8
    alpha_vis = (alpha_np.clip(0.0, 1.0) * 255.0).astype("uint8")
    alpha_img = Image.fromarray(alpha_vis, mode="L")

    # Depth: min-max normalize for visualization
    d_min, d_max = float(depth_np.min()), float(depth_np.max())
    if d_max <= d_min + 1e-6:
        depth_vis = np.zeros_like(depth_np, dtype="uint8")
    else:
        depth_vis = (255.0 * (depth_np - d_min) / (d_max - d_min + 1e-6)).clip(0, 255).astype("uint8")
    depth_img = Image.fromarray(depth_vis, mode="L")

    # 6. Save outputs
    os.makedirs(out_dir, exist_ok=True)
    stem = img_p.stem
    alpha_path = os.path.join(out_dir, f"{stem}_alpha.png")
    depth_path = os.path.join(out_dir, f"{stem}_depth.png")

    alpha_img.save(alpha_path)
    depth_img.save(depth_path)

    print(f"Saved alpha to {alpha_path}")
    print(f"Saved depth to {depth_path}")

    return alpha_np, depth_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to matte_depth_pretrained.pt")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input RGB image")
    parser.add_argument("--out_dir", type=str, default="./outputs_infer")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_inference(
        ckpt_path=args.ckpt,
        image_path=args.image,
        out_dir=args.out_dir,
        image_size=args.image_size,
        device=args.device,
    )
