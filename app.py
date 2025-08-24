import base64
import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
from typing import Optional
import uuid
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Background Removal API",
              description="API for removing image backgrounds using RMBG-1.4 model")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5500"] if serving HTML with Live Server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ID = "luffy1412/BG_Remove"   # replace with your repo name
MODEL_FILENAME = "model_fp16.onnx"

# Download and cache model automatically
MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)

# Initialize ONNX model
try:
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
except Exception as e:
    raise RuntimeError(f"Failed to load ONNX model: {str(e)}")

# Temporary directory for processed files
TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

def remove_background(img: np.ndarray) -> np.ndarray:
    """Core background removal function"""
    h, w = img.shape[:2]
    
    # Optimized processing based on image size
    if max(h, w) <= 512:
        inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = inp.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, :, :, :]
        pred = session.run(None, {input_name: inp})[0][0, 0]
        alpha = np.clip(pred, 0, 1)
    else:
        inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(inp, (1024, 1024)).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, :, :, :]
        pred = session.run(None, {input_name: inp})[0][0, 0]
        pred = cv2.resize(pred, (w, h))
        alpha = np.clip(pred, 0, 1)

    # Fast alpha smoothing
    if max(h, w) <= 512:
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    else:
        alpha = cv2.bilateralFilter((alpha * 255).astype(np.uint8), 9, 75, 75)
        alpha = alpha.astype(np.float32) / 255.0

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
    return rgba

import threading
import time

def schedule_file_deletion(file_path: str, delay: int = 60):
    """Delete file after a delay (in seconds)"""
    def delete_file():
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted temp file: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

    timer = threading.Timer(delay, delete_file)
    timer.start()


@app.post("/remove-bg/")
async def remove_background_endpoint(
    file: UploadFile = File(..., description="Image file to process"),
    return_format: Optional[str] = "png"
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        file_ext = os.path.splitext(file.filename)[1]
        temp_input = os.path.join(TEMP_DIR, f"input_{uuid.uuid4()}{file_ext}")
        temp_output = os.path.join(TEMP_DIR, f"output_{uuid.uuid4()}.png")

        with open(temp_input, "wb") as buffer:
            buffer.write(await file.read())

        img = cv2.imread(temp_input)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        result = remove_background(img)
        cv2.imwrite(temp_output, result)

        # Encode as base64
        with open(temp_output, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
            
        # Schedule deletion of output after 1 minute
        schedule_file_deletion(temp_output, delay=60)

        return JSONResponse(content={"image": img_base64})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_input):
            os.remove(temp_input)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 