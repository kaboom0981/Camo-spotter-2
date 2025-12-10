# backend/server.py
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from predict import (
    init_model, run_inference,
    make_overlay, make_bounding_box,
    make_heatmap, side_by_side
)
from PIL import Image

WEIGHTS_PATH = r"D:\Camo spotter 3\Camo-spotter-2\backend\weights\sinet.pth"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
model = init_model(WEIGHTS_PATH)
print("Model ready.")


def to_png_bytes(img_np):
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    original = Image.open(io.BytesIO(contents)).convert("RGB")

    # MASK
    mask = run_inference(model, original)

    # VISUALS
    overlay = make_overlay(original, mask)
    bbox = make_bounding_box(original, mask)
    heatmap = make_heatmap(mask)
    combined = side_by_side(original, mask, overlay)

    return {
        "mask": StreamingResponse(to_png_bytes(mask), media_type="image/png"),
        "overlay": StreamingResponse(to_png_bytes(overlay), media_type="image/png"),
        "bounding_box": StreamingResponse(to_png_bytes(bbox), media_type="image/png"),
        "heatmap": StreamingResponse(to_png_bytes(heatmap), media_type="image/png"),
        "combined": StreamingResponse(to_png_bytes(combined), media_type="image/png"),
    }

