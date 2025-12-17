import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class ResNetEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = resnet34(pretrained=pretrained)
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1 # now its 64
        self.layer2 = m.layer2 # 128
        self.layer3 = m.layer3 # 256
        self.layer4 = m.layer4 # 512

    def forward(self, x):
        x = self.conv1(x) # 1/2
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x) # 1/4
        x1 = self.layer1(x0) # 1/4, 64
        x2 = self.layer2(x1) # 1/8, 128
        x3 = self.layer3(x2) # 1/16, 256
        x4 = self.layer4(x3) # 1/32, 512
        return x1, x2, x3, x4 # skip features


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            skip = skip[:, :, dh // 2 : dh // 2 + x.shape[-2], dw // 2 : dw // 2 + x.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DecoderHead(nn.Module):
    def __init__(self, out_channels: int = 1):
        super().__init__()
        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, feats, out_size):
        x1, x2, x3, x4 = feats  # from encoder
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.conv_out(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x


class MatDepthNet(nn.Module):
    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained_encoder)
        self.alpha_head = DecoderHead(out_channels=1)
        self.depth_head = DecoderHead(out_channels=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = self.encoder(x)
        alpha = self.alpha_head(feats, (h, w))
        depth = self.depth_head(feats, (h, w))
        alpha = torch.sigmoid(alpha)  # [0,1]
        return alpha, depth