import os
import random
import shutil

FULL_IMAGES = r"D:\Camo spotter 3\camo-spotter-2\datasets\COD10K_full\Train\Images"
FULL_GT     = r"D:\Camo spotter 3\camo-spotter-2\datasets\COD10K_full\Train\GT_Object"

SUBSET_DIR = r"D:\Camo spotter 3\camo-spotter-2\datasets\COD10K_subset"
SUBSET_IMAGES = os.path.join(SUBSET_DIR, "Images")
SUBSET_GT = os.path.join(SUBSET_DIR, "GT")

os.makedirs(SUBSET_IMAGES, exist_ok=True)
os.makedirs(SUBSET_GT, exist_ok=True)

N = 500

# list all images
all_images = [
    f for f in os.listdir(FULL_IMAGES)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

random.seed(42)
random.shuffle(all_images)

count = 0

gt_files = set(os.listdir(FULL_GT))  # IMPORTANT: fast lookup!

for img in all_images:
    if count >= N:
        break

    name, ext = os.path.splitext(img)

    # masks may be .png OR .jpg
    possible_masks = [name + ".png", name + ".jpg"]

    mask = None
    for pm in possible_masks:
        if pm in gt_files:
            mask = pm
            break

    if mask is None:
        continue  # mask not found → skip!

    # copy both files
    shutil.copy(os.path.join(FULL_IMAGES, img), SUBSET_IMAGES)
    shutil.copy(os.path.join(FULL_GT, mask), SUBSET_GT)

    count += 1

print(f"Subset created: {count} image-mask pairs.")
