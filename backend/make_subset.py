# backend/make_subset.py
import os
import random
import shutil

FULL_IMAGES = "datasets/COD10K_full/Images"
FULL_GT = "datasets/COD10K_full/GT"

SUBSET_DIR = "datasets/COD10K_subset"
SUBSET_IMAGES = os.path.join(SUBSET_DIR, "Images")
SUBSET_GT = os.path.join(SUBSET_DIR, "GT")
N = 500  # number of images to pick

os.makedirs(SUBSET_IMAGES, exist_ok=True)
os.makedirs(SUBSET_GT, exist_ok=True)

all_images = [f for f in os.listdir(FULL_IMAGES) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

random.seed(42)
selected = random.sample(all_images, N)

count = 0
for fname in selected:
    img_path = os.path.join(FULL_IMAGES, fname)
    mask_path = os.path.join(FULL_GT, fname)

    # Ensure mask exists with same name
    if not os.path.exists(mask_path):
        continue

    shutil.copy(img_path, os.path.join(SUBSET_IMAGES, fname))
    shutil.copy(mask_path, os.path.join(SUBSET_GT, fname))
    count += 1

print(f"✅ Subset created: {count} image-mask pairs.")
print(f"Saved in: {SUBSET_DIR}")
