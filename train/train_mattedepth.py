import os
import argparse
import random

import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.mattedepth_net_poc import MatDepthNet
from datasets.aim500_depth import AIM500DepthDataset, collect_aim500_image_paths
from losses.matting_depth_losses import MultiTaskLoss

# use like : 
# python train_mat_depth.py --aim_root ./datasets/AIM-500 --image_size 512 --batch_size 4 --epochs 30 --device cuda --out_dir ./pretrained/matte_depth_pretrained.pt


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loss_fn, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        img = batch["image"]
        alpha_gt = batch["alpha"]
        depth_teacher = batch["depth_teacher"]

        optimizer.zero_grad(set_to_none=True)
        alpha_pred, depth_pred = model(img)
        loss_dict = loss_fn(alpha_pred, depth_pred, alpha_gt, depth_teacher)
        loss = loss_dict["total"]
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loss_fn, loader, device):
    model.eval()
    running_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        img = batch["image"]
        alpha_gt = batch["alpha"]
        depth_teacher = batch["depth_teacher"]

        alpha_pred, depth_pred = model(img)
        loss_dict = loss_fn(alpha_pred, depth_pred, alpha_gt, depth_teacher)
        running_loss += loss_dict["total"].item() * img.size(0)
    return running_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aim_root", type=str, required=True,
                        help="Path to AIM-500 root (containing original/, mask/, depthpro/)")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="checkpoints_aim500")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = args.device
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        device = "cpu"

    # collect all original images
    original_root = os.path.join(args.aim_root, "original")
    all_paths = collect_aim500_image_paths(original_root)
    if len(all_paths) == 0:
        raise RuntimeError(f"No images found under {original_root}")

    # train/val split
    indices = list(range(len(all_paths)))
    random.shuffle(indices)
    val_size = int(len(indices) * args.val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_paths = [all_paths[i] for i in train_indices]
    val_paths = [all_paths[i] for i in val_indices]

    train_ds = AIM500DepthDataset(
        root_dir=args.aim_root,
        image_paths=train_paths,
        image_size=args.image_size,
        augment=True,
    )
    val_ds = AIM500DepthDataset(
        root_dir=args.aim_root,
        image_paths=val_paths,
        image_size=args.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = MatDepthNet(pretrained_encoder=True).to(device)
    loss_fn = MultiTaskLoss(lambda_alpha=1.0, lambda_depth=0.5, lambda_edge=0.1)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loss_fn, train_loader, optimizer, device)
        val_loss = validate(model, loss_fn, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.out_dir, "matte_depth_pretrained.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            print(f"  -> saved {ckpt_path}")


if __name__ == "__main__":
    main()