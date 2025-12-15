import os
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return {k[len("module."):]: v for k, v in sd.items()}


def _load_state_dict(ckpt_path: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "net", "generator"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return _strip_module_prefix(ckpt[key])
        # sometimes it's already a raw state dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return _strip_module_prefix(ckpt)  # type: ignore
    raise RuntimeError(f"Unrecognized checkpoint format: {ckpt_path}")


class PCTNetRunner:
    """
    Minimal inference wrapper for PCTNet_CNN pretrained harmonizer.

    Expects:
      - model code at ./models/PCTNet_CNN.py
      - weights at ./pretrained_models/PCTNet_CNN.pt

    It builds low-res (256x256) and full-res (768x1024) inputs like the training config.
    """

    def __init__(
        self,
        ckpt_path: str = "./pretrained_models/PCTNet_CNN.pt",
        device: str = "cuda",
        crop_size: Tuple[int, int] = (768, 1024),  # (H,W)
        low_res_size: Tuple[int, int] = (256, 256),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        self.crop_size = crop_size
        self.low_res_size = low_res_size

        # Import your repo’s PCTNet + BMCONFIGS
        from models.PCTNet_CNN import PCTNet, BMCONFIGS  # noqa: F401

        ccfg = BMCONFIGS["CNN_pct"]
        self.model = PCTNet(**ccfg["params"]).to(self.device)
        self.model.eval()

        sd = _load_state_dict(ckpt_path, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if len(unexpected) > 0:
            print(f"[PCTNetRunner] warning: unexpected keys: {unexpected[:10]}")
        if len(missing) > 0:
            print(f"[PCTNetRunner] warning: missing keys: {missing[:10]}")

        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _norm(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01 - self.mean) / self.std

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.std + self.mean).clamp(0.0, 1.0)

    def _resize(self, x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def run(self, composite_01: torch.Tensor, mask_01: torch.Tensor) -> torch.Tensor:
        """
        composite_01: (B,3,H,W) in [0,1]
        mask_01:      (B,1,H,W) in [0,1] (use alpha, not just binary)

        Returns:
          harmonized_01_fullres: (B,3, crop_H, crop_W) in [0,1]
        """
        assert composite_01.ndim == 4 and composite_01.shape[1] == 3
        assert mask_01.ndim == 4 and mask_01.shape[1] == 1
        composite_01 = composite_01.to(self.device)
        mask_01 = mask_01.to(self.device)

        # Build low-res + full-res inputs
        comp_lr = self._resize(composite_01, self.low_res_size)
        mask_lr = self._resize(mask_01, self.low_res_size)

        comp_hr = self._resize(composite_01, self.crop_size)
        mask_hr = self._resize(mask_01, self.crop_size)

        comp_lr_n = self._norm(comp_lr)
        comp_hr_n = self._norm(comp_hr)

        batch = {
            "images": comp_lr_n,              # normalized
            "masks": mask_lr,                 # typically not normalized
            "images_fullres": comp_hr_n,      # normalized
            "masks_fullres": mask_hr,         # mask
        }

        # Try common forward signatures
        out = None
        try:
            out = self.model(batch)
        except TypeError:
            pass

        if out is None:
            try:
                out = self.model(comp_lr_n, mask_lr, comp_hr_n, mask_hr)
            except TypeError:
                pass

        if out is None:
            try:
                out = self.model(comp_hr_n, mask_hr)
            except TypeError as e:
                raise RuntimeError(
                    "Could not call PCTNet forward. Try inspecting PCTNet.forward signature and "
                    "adjusting PCTNetRunner.run() call pattern."
                ) from e

        # Extract predicted image tensor
        pred = out
        if isinstance(out, dict):
            for k in ["images_fullres", "pred_fullres", "output_fullres", "output", "pred", "images"]:
                if k in out:
                    pred = out[k]
                    break

        if not torch.is_tensor(pred):
            raise RuntimeError(f"PCTNet output type not understood: {type(pred)}")

        # pred is assumed normalized RGB (same space as inputs/targets)
        pred_01 = self._denorm(pred)
        return pred_01
