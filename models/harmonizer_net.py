# models/harmonizer_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class HarmonizerNet(nn.Module):
    """
    U-Net-like harmonizer.
    Inputs: composite (3), mask (1), optional depth_bg (1) -> C_in = 4 or 5
    Output: harmonized composite = composite + residual
    """

    def __init__(self, use_depth: bool = False, base_ch: int = 64):
        super().__init__()
        self.use_depth = use_depth
        in_ch = 3 + 1 + (1 if use_depth else 0)

        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=1)

    def forward(self, composite, mask, depth_bg=None):
        """
        composite: (B,3,H,W)
        mask:      (B,1,H,W)
        depth_bg:  (B,1,H,W) or None
        """
        if self.use_depth:
            assert depth_bg is not None, "use_depth=True but depth_bg is None"
            x = torch.cat([composite, mask, depth_bg], dim=1)
        else:
            x = torch.cat([composite, mask], dim=1)

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        b = self.bottleneck(p4)

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        residual = self.out_conv(d1)
        harmonized = composite + residual
        return harmonized.clamp(0.0, 1.0)
