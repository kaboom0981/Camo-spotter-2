# backend/utils.py
import io
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

# transforms for input image
def preprocess_image_pil(pil_image, target_size=352):
    # convert to RGB, resize maintaining aspect ratio and pad to square target
    img = pil_image.convert("RGB")
    # Resize while preserving aspect ratio to have max side = target_size
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

    # paste on black background to make target_size x target_size
    background = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    background.paste(img_resized, (left, top))

    # to tensor and normalize (imagenet)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(background).unsqueeze(0)  # [1,3,H,W]
    meta = {"orig_size": (w, h), "paste": (left, top, new_w, new_h), "target": target_size}
    return tensor, meta

def postprocess_mask(mask_tensor, meta, threshold=0.5):
    # mask_tensor: [1,1,H,W] values 0..1
    import numpy as np
    from PIL import Image
    mask = mask_tensor.squeeze().cpu().numpy()
    # crop back to original image area
    left, top, new_w, new_h = meta["paste"]
    target = meta["target"]
    # crop the center region corresponding to the resized image
    crop = mask[top:top+new_h, left:left+new_w]
    # resize crop to original size
    orig_w, orig_h = meta["orig_size"]
    crop_img = Image.fromarray((crop * 255).astype("uint8")).resize((orig_w, orig_h), Image.BILINEAR)
    # binary mask
    bin_mask = (np.array(crop_img) >= int(threshold * 255)).astype("uint8") * 255
    pil_mask = Image.fromarray(bin_mask).convert("L")
    return pil_mask

def mask_to_bytes(pil_mask, fmt="PNG"):
    bio = io.BytesIO()
    pil_mask.save(bio, format=fmt)
    bio.seek(0)
    return bio.getvalue()
