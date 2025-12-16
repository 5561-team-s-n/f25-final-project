import torch
import numpy as np
from PIL import Image
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation


class DepthProRunner:
    def __init__(self, device="cuda"):
        self.device = device
        self.processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
        self.model = DepthProForDepthEstimation.from_pretrained(
            "apple/DepthPro-hf", use_fov_model=False
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def run(self, pil_img: Image.Image, target_size):
        """
        pil_img: PIL RGB image
        target_size: (H, W) to resize depth to (must match matting output)
        returns: torch.Tensor of shape (1,1,H,W)
        """
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        post = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[target_size]
        )
        depth = post[0]["predicted_depth"]  # (H,W), torch

        # normalize per-image (relative depth)
        depth = depth.float()
        depth = (depth - depth.mean()) / (depth.std() + 1e-6)

        return depth.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)