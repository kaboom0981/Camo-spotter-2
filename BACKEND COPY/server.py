# backend/server.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from predict import init_model, run_inference
from PIL import Image
import io

app = FastAPI(title="SINet Camouflage Detection")

# Allow your frontend origin; '*' okay for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to weights
WEIGHTS_PATH = r"D:\Camo spotter 3\Camo-spotter-2\backend\weights\sinet.pth"

# Initialize model at startup
print("Initializing model. This may take a few seconds...")
model = init_model(WEIGHTS_PATH)
print("Model loaded.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    mask_pil = run_inference(model, pil_img, threshold=0.5)
    # Return PNG bytes
    buf = io.BytesIO()
    mask_pil.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})
