# backend/train.py
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from dataset import COD10KDataset
from models.sinet import get_model

DEVICE = torch.device("cpu")  # You have Intel Iris → CPU only

DATASET_PATH = "datasets/COD10K_subset"
IMG_DIR = os.path.join(DATASET_PATH, "Images")
GT_DIR = os.path.join(DATASET_PATH, "GT")

WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
SAVE_PATH = os.path.join(WEIGHTS_DIR, "sinet.pth")

EPOCHS = 10          # You can increase to 20 later
BATCH_SIZE = 3       # Safe for CPU
LR = 1e-4            # Learning rate

def bce_loss(pred, gt):
    return F.binary_cross_entropy(pred, gt)

def train():
    print("📌 Loading dataset...")
    ds = COD10KDataset(IMG_DIR, GT_DIR)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    print("📌 Initializing model...")
    model = get_model(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("🚀 Training started...\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch}/{EPOCHS}")
        for img, mask in pbar:
            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            optimizer.zero_grad()

            Ci, Cs = model(img)

            loss_ci = bce_loss(Ci, mask)
            loss_cs = bce_loss(Cs, mask)

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
