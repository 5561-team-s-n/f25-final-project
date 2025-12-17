import os
import cv2
import torch
import logging
import numpy as np
from utils.matteformer_config import CONFIG
import torch.distributed as dist
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import torch.nn.functional as F

def make_dir(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


def print_network(model, name):
    """
    Print out the network information
    """
    logger = logging.getLogger("Logger")
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()

    logger.info(model)
    logger.info(name)
    logger.info("Number of parameters: {}".format(num_params))


def update_lr(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_lr(init_lr, step, iter_num):
    return step/iter_num*init_lr


def remove_prefix_state_dict(state_dict, prefix="module"):
    """
    remove prefix from the key of pretrained state dict for Data-Parallel
    """
    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key[len(prefix)+1:]] = state_dict[key].float()
    return new_state_dict


def get_unknown_tensor(trimap):
    """
    get 1-channel unknown area tensor from the 3-channel/1-channel trimap tensor
    """
    if CONFIG.model.trimap_channel == 3:
        weight = trimap[:, 1:2, :, :].float()
    else:
        weight = trimap.eq(1).float()
    return weight


def reduce_tensor_dict(tensor_dict, mode='mean'):
    """
    average tensor dict over different GPUs
    """
    for key, tensor in tensor_dict.items():
        if tensor is not None:
            tensor_dict[key] = reduce_tensor(tensor, mode)
    return tensor_dict


def reduce_tensor(tensor, mode='mean'):
    """
    average tensor over different GPUs
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if mode == 'mean':
        rt /= CONFIG.world_size
    elif mode == 'sum':
        pass
    else:
        raise NotImplementedError("reduce mode can only be 'mean' or 'sum'")
    return rt


Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]


def get_unknown_tensor_from_pred(pred, rand_width=30, train_mode=True):
    ### pred: N, 1 ,H, W 
    N, C, H, W = pred.shape
    pred = F.interpolate(pred, size=(640,640), mode='nearest')
    pred = pred.data.cpu().numpy()
    uncertain_area = np.ones_like(pred, dtype=np.uint8)
    uncertain_area[pred < 1.0/255.0] = 0
    uncertain_area[pred > 1-1.0/255.0] = 0

    for n in range(N):
        uncertain_area_ = uncertain_area[n,0,:,:] # H, W
        if train_mode:
            width = np.random.randint(1, rand_width)
        else:
            width = rand_width // 2
        uncertain_area_ = cv2.dilate(uncertain_area_, Kernels[width])
        uncertain_area[n,0,:,:] = uncertain_area_

    weight = np.zeros_like(uncertain_area)
    weight[uncertain_area == 1] = 1

    weight = np.array(weight, dtype=float)
    weight = torch.from_numpy(weight).cuda()

    weight = F.interpolate(weight, size=(H,W), mode='nearest')

    return weight

def pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def normalize_imagenet(x01: torch.Tensor, device: str) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=x01.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - mean) / std


def tensor_rgb_to_pil(x01: torch.Tensor) -> Image.Image:
    x = x01.detach().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    x = (x * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


def tensor_gray_to_pil(x01: torch.Tensor) -> Image.Image:
    x = x01.detach().clamp(0.0, 1.0).squeeze(0).squeeze(0).cpu().numpy()
    x = (x * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


def _depth_to_uint8(depth: torch.Tensor, eps: float = 1e-6) -> np.ndarray:
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

    ticks = [1.0, 0.75, 0.5, 0.25, 0.0]
    for p in ticks:
        y = int(round(bar_y0 + (1.0 - p) * (bar_h - 1)))
        draw.line([(bar_x0 + bar_w + 2, y), (bar_x0 + bar_w + 8, y)], fill=(0, 0, 0))
        val = d_min + p * (d_max - d_min)
        label = f"{val:.3g}" + (f" {unit}" if unit else "")
        draw.text((bar_x0 + bar_w + 12, y - 6), label, fill=(0, 0, 0), font=font)

    return base


def save_depth(depth: torch.Tensor, out_base: Path, debug: bool):
    out_base.parent.mkdir(parents=True, exist_ok=True)

    d = depth.detach().squeeze(0).squeeze(0).float().cpu().numpy().astype(np.float32)
    # np.save(out_base.with_suffix(".npy"), d)

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

def get_masked_local_from_global(global_sigmoid, local_sigmoid):
	values, index = torch.max(global_sigmoid,1)
	index = index[:,None,:,:].float()
	### index <===> [0, 1, 2]
	### bg_mask <===> [1, 0, 0]
	bg_mask = index.clone()
	bg_mask[bg_mask==2]=1
	bg_mask = 1- bg_mask
	### trimap_mask <===> [0, 1, 0]
	trimap_mask = index.clone()
	trimap_mask[trimap_mask==2]=0
	### fg_mask <===> [0, 0, 1]
	fg_mask = index.clone()
	fg_mask[fg_mask==1]=0
	fg_mask[fg_mask==2]=1
	fusion_sigmoid = local_sigmoid*trimap_mask+fg_mask
	return fusion_sigmoid

#MATTEFORMER helpers
def gen_trimap(alpha, max_kernel_size=30):
    """
    Generate a trimap using MatteFormer's erosion method.
    alpha: uint8 grayscale alpha matte (0-255)
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
        weight_os4 = get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width1, train_mode=False)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        
        weight_os1 = get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width2, train_mode=False)
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

