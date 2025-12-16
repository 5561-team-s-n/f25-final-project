# utils/occlusion_composite.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def _resize(x: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False if mode == "bilinear" else None)

def _depth_normalize(d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Per-image (per-batch element) z-normalization.
    # d must be like (B,1,H,W)
    mean = d.mean(dim=(2, 3), keepdim=True)
    std = d.std(dim=(2, 3), keepdim=True)
    return (d - mean) / (std + eps)

def _soft_composite(depth_bg: torch.Tensor, depth_fg: torch.Tensor, sharpness: float) -> torch.Tensor:
    return torch.sigmoid((depth_bg - depth_fg) * sharpness)

def depth_aware_composite(
    fg_rgb: torch.Tensor, # (B,3,H,W) 
    alpha: torch.Tensor, # (B,1,H,W) 
    depth_fg: torch.Tensor, # (B,1,H,W)
    bg_rgb: torch.Tensor, # (B,3,H,W) 
    depth_bg: torch.Tensor, # (B,1,H,W) 
    depth_shift: float, # positive pushes fg farther, negative pulls fg closer
    *,
    sharpness: float = 10.0, # higher means harder composite boundary
    use_soft_composite: bool = True, # if false uses hard compare where a closer background just fully occludes fg with no smoothing
    clamp: bool = True,
):
    # Composites foreground onto background using alpha + depth-based occlusion.
    # Assumptions: smaller depth values mean "closer" 
    assert fg_rgb.ndim == 4 and fg_rgb.shape[1] == 3
    assert bg_rgb.ndim == 4 and bg_rgb.shape[1] == 3
    assert alpha.ndim == 4 and alpha.shape[1] == 1
    assert depth_fg.ndim == 4 and depth_fg.shape[1] == 1
    assert depth_bg.ndim == 4 and depth_bg.shape[1] == 1
    assert fg_rgb.shape[0] == bg_rgb.shape[0] == alpha.shape[0] == depth_fg.shape[0] == depth_bg.shape[0]

    # Ensure shapes match
    alpha = _resize(alpha, fg_rgb, mode="bilinear")
    depth_fg = _resize(depth_fg, fg_rgb, mode="bilinear")
    depth_bg = _resize(depth_bg, fg_rgb, mode="bilinear")
    bg_rgb = _resize(bg_rgb, fg_rgb, mode="bilinear")

    # Clean alpha
    alpha = alpha.clamp(0.0, 1.0)

    # Normalize depths per-image 
    df = _depth_normalize(depth_fg)
    db = _depth_normalize(depth_bg)

    # Apply shift 
    df = df + depth_shift

    # gate=1 means fg is visible (in front), gate=0 means bg occludes
    if use_soft_composite:
        gate = _soft_composite(db, df, sharpness)  # (B,1,H,W)
    else:
        # hard composite: fg visible where it is closer than bg
        gate = (df < db).float()

    # Effective alpha after occlusion
    new_fg_alpha = alpha * gate

    # Standard alpha composite
    out = fg_rgb * new_fg_alpha + bg_rgb * (1.0 - new_fg_alpha)

    if clamp:
        out = out.clamp(0.0, 1.0)
    return out, new_fg_alpha
