import torch
import torch.nn.functional as F
import torch.nn as nn


def _sobel_kernels(device):
    gx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3) / 8.0
    gy = gx.transpose(2, 3).contiguous()
    return gx, gy


def gradient_mag(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,1,H,W)
    """
    gx_k, gy_k = _sobel_kernels(x.device)
    gx = F.conv2d(x, gx_k, padding=1)
    gy = F.conv2d(x, gy_k, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def matting_loss(alpha_pred: torch.Tensor, alpha_gt: torch.Tensor, w_grad: float = 0.5):
    """
    alpha_pred, alpha_gt: (B,1,H,W), values in [0,1]
    """
    l1 = F.l1_loss(alpha_pred, alpha_gt)
    grad_pred = gradient_mag(alpha_pred)
    grad_gt = gradient_mag(alpha_gt)
    grad_l1 = F.l1_loss(grad_pred, grad_gt)
    return l1 + w_grad * grad_l1


def _normalize_per_image(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,1,H,W) -> per-image z-normalized
    """
    mean = x.mean(dim=[2, 3], keepdim=True)
    std = x.std(dim=[2, 3], keepdim=True) + 1e-6
    return (x - mean) / std


def depth_loss(depth_pred: torch.Tensor, depth_teacher: torch.Tensor):
    """
    depth_pred, depth_teacher: (B,1,H,W), arbitrary scale.
    Use per-image z-normalization to enforce relative structure matching.
    """
    depth_pred_n = _normalize_per_image(depth_pred)
    depth_teacher_n = _normalize_per_image(depth_teacher)
    return F.l1_loss(depth_pred_n, depth_teacher_n)


def edge_consistency_loss(
    alpha_pred: torch.Tensor,
    depth_pred: torch.Tensor,
    alpha_gt: torch.Tensor,
    band_low: float = 0.1,
    band_high: float = 0.9,
    eps: float = 1e-6,
):
    """
    Encourage depth and alpha edges to align in semi-transparent boundary band.
    """
    # normalize depth before gradients
    depth_pred_n = _normalize_per_image(depth_pred)

    grad_alpha = gradient_mag(alpha_pred)
    grad_depth = gradient_mag(depth_pred_n)

    # normalize gradient magnitudes to [0,1] per image
    def _norm_grad(g):
        g_min = g.amin(dim=[2, 3], keepdim=True)
        g_max = g.amax(dim=[2, 3], keepdim=True)
        return (g - g_min) / (g_max - g_min + eps)

    grad_alpha_n = _norm_grad(grad_alpha)
    grad_depth_n = _norm_grad(grad_depth)

    band = (alpha_gt > band_low) & (alpha_gt < band_high)
    if band.sum() == 0:
        return torch.tensor(0.0, device=alpha_pred.device, dtype=alpha_pred.dtype)

    diff = torch.abs(grad_alpha_n - grad_depth_n)
    diff = diff * band.float()
    return diff.sum() / (band.sum() + eps)


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        lambda_alpha: float = 1.0,
        lambda_depth: float = 0.5,
        lambda_edge: float = 0.1,
        grad_weight: float = 0.5,
    ):
        super().__init__()
        self.lambda_alpha = lambda_alpha
        self.lambda_depth = lambda_depth
        self.lambda_edge = lambda_edge
        self.grad_weight = grad_weight

    def forward(self, alpha_pred, depth_pred, alpha_gt, depth_teacher):
        l_alpha = matting_loss(alpha_pred, alpha_gt, w_grad=self.grad_weight)
        l_depth = depth_loss(depth_pred, depth_teacher)
        l_edge = edge_consistency_loss(alpha_pred, depth_pred, alpha_gt)
        total = (
            self.lambda_alpha * l_alpha
            + self.lambda_depth * l_depth
            + self.lambda_edge * l_edge
        )
        return {
            "total": total,
            "alpha": l_alpha.detach(),
            "depth": l_depth.detach(),
            "edge": l_edge.detach(),
        }
