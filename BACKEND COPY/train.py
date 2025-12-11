# backend/train.py
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from dataset import COD10KDataset
from models.sinet import get_model

# Use CPU
DEVICE = torch.device("cpu")

# Optional: slightly limit PyTorch threads so it doesn’t thrash your CPU
torch.set_num_threads(4)

DATASET_PATH = r"D:\Camo spotter 3\Camo-spotter-2\datasets\COD10K_subset"

IMG_DIR = os.path.join(DATASET_PATH, "Images")
GT_DIR = os.path.join(DATASET_PATH, "GT")

WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
SAVE_PATH = os.path.join(WEIGHTS_DIR, "sinet.pth")

EPOCHS = 5           # start with 5; you can bump to 10 later if okay
BATCH_SIZE = 3       # safe on CPU
LR = 1e-4            # learning rate
MAX_SAMPLES = 300    # use 300 images for faster training; increase if you want


def bce_loss(pred, gt):
    return F.binary_cross_entropy(pred, gt)


def train():
    print("📌 Loading dataset...")
    ds = COD10KDataset(IMG_DIR, GT_DIR, target_size=352, max_samples=MAX_SAMPLES)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,     # you can try 2 or 4 if your CPU handles it
        pin_memory=False
    )

    print(f"✅ Dataset size: {len(ds)} images")
    print("📌 Initializing model...")
    model = get_model(device=DEVICE)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    print("🚀 Training started...\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch}/{EPOCHS}")
        for img, mask in pbar:
            img = img.to(DEVICE)      # [B,3,H,W]
            mask = mask.to(DEVICE)    # [B,1,H,W]

            optimizer.zero_grad()

            Ci, Cs = model(img)       # both [B,1,H,W] now

            # Resize GT to match output, just in case
            mask_ci = F.interpolate(mask, size=Ci.shape[-2:], mode="nearest")
            mask_cs = F.interpolate(mask, size=Cs.shape[-2:], mode="nearest")

            loss_ci = bce_loss(Ci, mask_ci)
            loss_cs = bce_loss(Cs, mask_cs)

            loss = loss_ci + 0.5 * loss_cs

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch} complete. Avg Loss = {avg_loss:.4f}")

        torch.save({"state_dict": model.state_dict()}, SAVE_PATH)

    print("\n🎉 Training finished!")
    print(f"💾 Model saved to: {SAVE_PATH}")


if __name__ == "__main__":
    train()
