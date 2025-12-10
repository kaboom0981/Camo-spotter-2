# backend/predict.py
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
from models.sinet import load_model

transform = T.Compose([
    T.Resize((352, 352)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def init_model(weights_path, device="cpu"):
    return load_model(weights_path, device=device)

def run_inference(model, pil_img, threshold=0.5):
    model.eval()

    img_tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        Ci, Cs = model(img_tensor)

    mask = Ci.squeeze().cpu().numpy()
    mask = (mask > threshold).astype(np.uint8) * 255

    return mask


# ------------------ EXTRA VISUALIZATIONS ------------------

def make_overlay(original, mask):
    orig = np.array(original)
    mask_colored = np.zeros_like(orig)
    mask_colored[:, :, 2] = mask  # Red channel

    overlay = cv2.addWeighted(orig, 0.7, mask_colored, 0.3, 0)
    return overlay


def make_bounding_box(original, mask):
    orig = np.array(original).copy()
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return orig  # No detection

    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return orig


def make_heatmap(mask):
    heat = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return heat


def side_by_side(original, mask, overlay):
    orig = np.array(original)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    combined = np.hstack((orig, mask_rgb, overlay))
    return combined
