from __future__ import annotations

from pathlib import Path
import torch
import torch.nn.functional as F

from kornia.color import RgbToHsv, HsvToRgb
from iharm.model.base.harmonizer_net import Harmonize
from iharm.mconfigs import BMCONFIGS


def _pick_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "generator"):
            v = ckpt_obj.get(k, None)
            if isinstance(v, dict):
                return v
    return ckpt_obj


class HarmonizeRunner:
    def __init__(self, ckpt_path: str, device: str):
        ccfg = BMCONFIGS["ViT_aict"]
        params = dict(ccfg["params"])
        params.setdefault("input_normalization", {"mean": [0, 0, 0], "std": [1, 1, 1]})

        self.color_space = str(ccfg.get("data", {}).get("color_space", "RGB")).upper()
        self.device = device

        self.model = Harmonize(**params).to(device).eval()

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        sd = _pick_state_dict(ckpt)
        self.model.load_state_dict(sd, strict=False)

        if self.color_space == "HSV":
            self._rgb2hsv = RgbToHsv().to(device)
            self._hsv2rgb = HsvToRgb().to(device)
            self._hsv_scale = torch.tensor([6.283, 1.0, 1.0], device=device).view(1, 3, 1, 1)
        else:
            self._rgb2hsv = None
            self._hsv2rgb = None
            self._hsv_scale = None

    @torch.no_grad()
    def run(self, comp_rgb01: torch.Tensor, mask01: torch.Tensor) -> torch.Tensor:
        x = comp_rgb01.to(self.device).clamp(0.0, 1.0)
        m = mask01.to(self.device).clamp(0.0, 1.0)

        if m.shape[-2:] != x.shape[-2:]:
            m = F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # color-space preprocessing
        if self._rgb2hsv is not None:
            x_full = self._rgb2hsv(x) / self._hsv_scale
        else:
            x_full = x
        m_full = m

        # low-res branch inputs for the model
        low_hw = (256, 256)
        x_low = F.interpolate(x_full, size=low_hw, mode="bilinear", align_corners=False)
        m_low = F.interpolate(m_full, size=low_hw, mode="bilinear", align_corners=False)

        out = self.model(
            image=x_low,
            image_fullres=x_full,
            mask=m_low,
            mask_fullres=m_full,
        )

        if isinstance(out, dict):
            if "images_fullres" in out and out["images_fullres"] is not None:
                y = out["images_fullres"]
            elif "images" in out and out["images"] is not None:
                y = out["images"]
            elif "output" in out and out["output"] is not None:
                y = out["output"]
            else:
                y = next(iter(out.values()))
        elif isinstance(out, (tuple, list)):
            y = out[0]
        else:
            y = out
        if self._hsv2rgb is not None:
            y = self._hsv2rgb(y * self._hsv_scale)

        return y.clamp(0.0, 1.0)