# utils/occlusion_composite.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def _resize_like(x: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode=mode, align_corners=False if mode == "bilinear" else None)

def _z_norm_per_image(d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Per-image (per-batch element) z-normalization.
    d: (B,1,H,W)
    """
    mean = d.mean(dim=(2, 3), keepdim=True)
    std = d.std(dim=(2, 3), keepdim=True)
    return (d - mean) / (std + eps)

# def _z_norm_weighted(d: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     """
#     Weighted per-image z-normalization.
#     d,w: (B,1,H,W), with w in [0,1] (e.g., alpha).
#     """
#     w = w.clamp(0.0, 1.0)
#     wsum = w.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
#     mean = (d * w).sum(dim=(2, 3), keepdim=True) / wsum
#     var = (w * (d - mean).pow(2)).sum(dim=(2, 3), keepdim=True) / wsum
#     std = (var + eps).sqrt()
#     return (d - mean) / std


# def _soft_occlusion_gate(depth_bg: torch.Tensor, depth_fg: torch.Tensor, sharpness: float) -> torch.Tensor:
#     """
#     Returns a [0,1] gate , foreground in front = 1
#     """
#     return torch.sigmoid((depth_bg - depth_fg) * sharpness)

def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    # x: (B,1,H,W)
    t = (x - edge0) / (edge1 - edge0 + 1e-12)
    t = t.clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def _soft_occlusion_gate(
    depth_bg: torch.Tensor,
    depth_fg: torch.Tensor,
    alpha: torch.Tensor,
    sharpness: float,
    *,
    alpha_edge0: float = 0.2, # below this, alpha dominates
    alpha_edge1: float = 0.95, # above this, depth dominates
    edge_exponent: float = 0.35, # <1 => relax depth gating at wispy edges
    occlusion_margin: float = 0.5, # in normalized depth units (std devs)
    occlusion_margin_sharpness: float = 20.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    # gate in [0,1], where 1 => FG visible, 0 => BG occludes.
    #   - compute depth gate p = sigmoid((db - df) * sharpness)
    #   - near wispy alpha edges, soften occlusion by using p**exp, exp<1
    #   - but if BG is *much* closer (strong evidence), don't soften (exp -> 1)
    delta = depth_bg - depth_fg # (B,1,H,W); >0 => fg closer
    p = torch.sigmoid(delta * sharpness) # base depth gate

    w_interior = _smoothstep(alpha_edge0, alpha_edge1, alpha) # 0=edge, 1=interior
    exp_alpha = edge_exponent + (1.0 - edge_exponent) * w_interior

    # 0 when BG is strongly closer (delta << -margin), 1 otherwise
    relax = torch.sigmoid((delta + occlusion_margin) * occlusion_margin_sharpness)
    exp = 1.0 + (exp_alpha - 1.0) * relax

    return p.clamp(min=eps, max=1.0).pow(exp)


def depth_aware_composite(
    fg_rgb: torch.Tensor, # (B,3,H,W) in [0,1] (foreground image or cutout)
    alpha: torch.Tensor, # (B,1,H,W) in [0,1] (foreground alpha)
    depth_fg: torch.Tensor, # (B,1,H,W) (predicted depth for fg or teacher depth)
    bg_rgb: torch.Tensor, # (B,3,H,W) in [0,1]
    depth_bg: torch.Tensor, # (B,1,H,W) (DepthPro depth for background)
    *,
    depth_shift: float = 1.0, # positive pushes fg farther, negative pulls fg closer
    sharpness: float = 10.0, # higher => harder occlusion boundary
    use_soft_gate: bool = True, # if False uses hard compare (bg closer => occlude)
    clamp: bool = True,
) -> torch.Tensor:
    # Composites foreground onto background using alpha + depth-based occlusion.
    # Assumptions / conventions:
    #   - depth maps are comparable after per-image z-normalization (relative depth).
    #   - smaller depth values mean "closer" 
    # Output:
    #   (B,3,H,W) composite in [0,1] (if clamp=True).
    assert fg_rgb.ndim == 4 and fg_rgb.shape[1] == 3
    assert bg_rgb.ndim == 4 and bg_rgb.shape[1] == 3
    assert alpha.ndim == 4 and alpha.shape[1] == 1
    assert depth_fg.ndim == 4 and depth_fg.shape[1] == 1
    assert depth_bg.ndim == 4 and depth_bg.shape[1] == 1
    assert fg_rgb.shape[0] == bg_rgb.shape[0] == alpha.shape[0] == depth_fg.shape[0] == depth_bg.shape[0]

    # Ensure shapes match
    alpha = _resize_like(alpha, fg_rgb, mode="bilinear")
    depth_fg = _resize_like(depth_fg, fg_rgb, mode="bilinear")
    depth_bg = _resize_like(depth_bg, fg_rgb, mode="bilinear")
    bg_rgb = _resize_like(bg_rgb, fg_rgb, mode="bilinear")

    # Clean alpha
    alpha = alpha.clamp(0.0, 1.0)

    # Normalize depths per-image (so they are at least comparable in scale)
    df = _z_norm_per_image(depth_fg)
    db = _z_norm_per_image(depth_bg)

    # Apply optional shift (in normalized units)
    df = df + depth_shift

    # Occlusion gating: gate=1 means fg is visible (in front), gate=0 means bg occludes
    if use_soft_gate:
        gate = _soft_occlusion_gate(db, df, alpha, sharpness=sharpness)  # (B,1,H,W)
    else:
        # hard z-buffer: fg visible where it is closer than bg
        # fg closer => df < db (if smaller depth is closer)
        gate = (df < db).float()

    # Effective alpha after occlusion
    alpha_eff = alpha * gate

    # Standard alpha composite
    out = fg_rgb * alpha_eff + bg_rgb * (1.0 - alpha_eff)

    if clamp:
        out = out.clamp(0.0, 1.0)
    return out
