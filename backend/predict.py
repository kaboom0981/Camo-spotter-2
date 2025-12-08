# backend/predict.py
import torch
from PIL import Image
from sinet import load_model, SINet
from utils import preprocess_image_pil, postprocess_mask, mask_to_bytes
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def init_model(weights_path=None):
    model = load_model(weights_path=weights_path, device=DEVICE)
    return model

def run_inference(model, pil_image, threshold=0.5):
    tensor, meta = preprocess_image_pil(pil_image, target_size=352)
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        Ci, Cs = model(tensor)   # Ci is final mask, Cs is coarse
    # choose Ci (final)
    mask_pil = postprocess_mask(Ci[0, 0], meta, threshold=threshold)
    return mask_pil
