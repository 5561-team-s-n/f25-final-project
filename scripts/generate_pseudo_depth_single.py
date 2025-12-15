# generate_pseudo_depth_single.py
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

# ---- config ----
IMAGE_NAME = "input.png"   # image in this directory
# -----------------

here = Path(__file__).parent
image_path = here / IMAGE_NAME

# load image
img = Image.open(image_path).convert("RGB")

# load depth model
processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
model = DepthProForDepthEstimation.from_pretrained(
    "apple/DepthPro-hf", use_fov_model=False
).to("cuda" if torch.cuda.is_available() else "cpu")
device = next(model.parameters()).device

# prepare & run
inputs = processor(images=img, return_tensors="pt").to(device)
with torch.no_grad():
    out = model(**inputs)
post = processor.post_process_depth_estimation(
    out, target_sizes=[(img.height, img.width)]
)
depth = post[0]["predicted_depth"].cpu().numpy().astype("float32")

# save npy
out_base = here / image_path.stem
np.save(out_base.with_name(image_path.stem + "_depth_full.npy"), depth)

# save PNG vis
d = depth
d_min, d_max = float(d.min()), float(d.max())
if d_max <= d_min + 1e-6:
    vis = np.zeros_like(d, dtype="uint8")
else:
    vis = ((d - d_min) / (d_max - d_min) * 255).clip(0, 255).astype("uint8")
Image.fromarray(vis).save(out_base.with_name(image_path.stem + "_depth_full.png"))

print("Saved:", image_path.stem + "_depth_full.npy")
print("Saved:", image_path.stem + "_depth_full.png")
