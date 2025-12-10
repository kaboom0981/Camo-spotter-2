# backend/models/sinet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# -----------------------------------------
#  Receptive Field Block (RF)
# -----------------------------------------
class RFBlock(nn.Module):
    """
    Simplified multi-branch RF module inspired by SINet paper.
    Each branch uses different kernel/dilation for large receptive fields.
    """
    def __init__(self, in_ch, out_ch=32):
        super().__init__()

        # Branch 1: 1x1 conv
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 3x3 conv
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 3: dilated conv (like 5x5)
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )

        # Branch 4: dilated conv (like 7x7)
        self.b4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=3, dilation=3),
            nn.ReLU(inplace=True)
        )

        # Fuse
        self.fuse = nn.Conv2d(out_ch * 4, out_ch, 1)

    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)
        fused = torch.cat([b1, b2, b3, b4], dim=1)
        return self.fuse(fused)


# -----------------------------------------
#  Partial Decoder Component (PDC)
# -----------------------------------------
class PDC(nn.Module):
    """
    Fuses multi-scale RF features into a single-channel camouflage map.
    """
    def __init__(self, in_ch_list):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(ch, 32, 3, padding=1) for ch in in_ch_list
        ])
        self.final = nn.Conv2d(32 * len(in_ch_list), 1, 1)

    def forward(self, feats, out_size):
        """
        feats: list of feature maps
        out_size: (H, W) of the desired output (usually input image size)
        """
        H, W = out_size
        ups = []

        for i, f in enumerate(feats):
            x = self.convs[i](f)
            x = F.interpolate(x, (H, W), mode="bilinear", align_corners=False)
            ups.append(x)

        x = torch.cat(ups, dim=1)
        x = self.final(x)
        return torch.sigmoid(x)


# -----------------------------------------
#                 SINet
# -----------------------------------------
class SINet(nn.Module):
    """
    Simplified SINet architecture with:
    - ResNet18 backbone (faster than ResNet50 – good for your CPU)
    - RF blocks
    - Multi-scale PDC fusion

    Produces: Ci (refined final map), Cs (coarse map)
    Both are [B,1,H,W] in SAME spatial size as input.
    """
    def __init__(self):
        super().__init__()

        # Use ResNet18 for speed
        res = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # ResNet18 structure:
        # res.conv1 -> res.bn1 -> res.relu -> res.maxpool
        # -> layer1 -> layer2 -> layer3 -> layer4
        self.stem = nn.Sequential(
            res.conv1,
            res.bn1,
            res.relu,
            res.maxpool,
        )
        self.layer1 = res.layer1  # 64
        self.layer2 = res.layer2  # 128
        self.layer3 = res.layer3  # 256
        self.layer4 = res.layer4  # 512

        # RF blocks (channel sizes match ResNet18)
        self.rf1 = RFBlock(64)   # layer1
        self.rf2 = RFBlock(128)  # layer2
        self.rf3 = RFBlock(256)  # layer3
        self.rf4 = RFBlock(512)  # layer4

        # Coarse PDC (use all RF features)
        self.pdc_s = PDC([32, 32, 32, 32])

        # Refined PDC (use deeper RF features only)
        self.pdc_i = PDC([32, 32, 32])

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.stem(x)    # 1/4 size
        x1 = self.layer1(x) # 64, 1/4
        x2 = self.layer2(x1) # 128, 1/8
        x3 = self.layer3(x2) # 256, 1/16
        x4 = self.layer4(x3) # 512, 1/32

        # RF features
        f1 = self.rf1(x1)
        f2 = self.rf2(x2)
        f3 = self.rf3(x3)
        f4 = self.rf4(x4)

        # Coarse map: use all RF features
        Cs = self.pdc_s([f1, f2, f3, f4], out_size=(H, W))

        # Refined map: use deeper features only
        Ci = self.pdc_i([f2, f3, f4], out_size=(H, W))

        return Ci, Cs


# -----------------------------------------
#   get_model() + load_model()
# -----------------------------------------
def get_model(device="cpu"):
    model = SINet().to(device)

    # OPTIONAL SPEED TRICK: freeze backbone (only RF + PDC learn)
    for p in model.stem.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    for p in model.layer3.parameters():
        p.requires_grad = False
    for p in model.layer4.parameters():
        p.requires_grad = False

    return model


def load_model(weights_path=None, device="cpu"):
    model = get_model(device=device)

    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)

    model.eval()
    return model
