# train_harmonizer.py
import os
import argparse
import random

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from datasets.iharmony4 import IHarmony4Dataset
from models.harmonizer_net import HarmonizerNet


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def l1_foreground_loss(pred, target, mask):
    """
    pred, target: (B,3,H,W)
    mask:        (B,1,H,W), [0,1]
    """
    w = mask
    return (torch.abs(pred - target) * w).sum() / (w.sum() * 3.0 + 1e-6)


def train_epoch(model, loader, optimizer, device, use_depth):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = to_device(batch, device)
        comp = batch["composite"]
        mask = batch["mask"]
        real = batch["real"]
        depth_bg = batch.get("depth_bg", None)

        optimizer.zero_grad(set_to_none=True)
        harmonized = model(comp, mask, depth_bg if use_depth else None)

        # L1 on full image + foreground-weighted L1
        loss_full = F.l1_loss(harmonized, real)
        loss_fg = l1_foreground_loss(harmonized, real, mask)
        loss = loss_full + loss_fg

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * comp.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device, use_depth):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = to_device(batch, device)
        comp = batch["composite"]
        mask = batch["mask"]
        real = batch["real"]
        depth_bg = batch.get("depth_bg", None)

        harmonized = model(comp, mask, depth_bg if use_depth else None)

        loss_full = F.l1_loss(harmonized, real)
        loss_fg = l1_foreground_loss(harmonized, real, mask)
        loss = loss_full + loss_fg
        total_loss += loss.item() * comp.size(0)

    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ih_root", type=str, required=True,
                        help="Root of iHarmony4 dataset")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--use_depth", action="store_true",
                        help="If true, expects depth_path column and uses depth_bg channel")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="checkpoints_harmonizer")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = args.device
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        device = "cpu"

    train_ds = IHarmony4Dataset(
        csv_path=args.train_csv,
        ih_root=args.ih_root,
        image_size=args.image_size,
        use_depth=args.use_depth,
        augment=True,
    )
    val_ds = IHarmony4Dataset(
        csv_path=args.val_csv,
        ih_root=args.ih_root,
        image_size=args.image_size,
        use_depth=args.use_depth,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = HarmonizerNet(use_depth=args.use_depth, base_ch=64).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.use_depth)
        val_loss = validate(model, val_loader, device, args.use_depth)
        scheduler.step()

        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.out_dir, "harmonizer_best.pt")
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
