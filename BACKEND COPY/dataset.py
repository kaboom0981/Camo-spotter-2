# backend/dataset.py
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class COD10KDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=352, max_samples=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size

        images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        masks = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        assert len(images) == len(masks), \
            "❌ Number of images and masks do not match!"

        # Optionally use only first N samples (speed!)
        if max_samples is not None:
            images = images[:max_samples]
            masks = masks[:max_samples]

        self.images = images
        self.masks = masks

        self.img_transform = T.Compose([
            T.Resize((target_size, target_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = self.masks[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.img_transform(img)

        mask = mask.resize((self.target_size, self.target_size), Image.NEAREST)
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1,H,W]

        return img, mask
