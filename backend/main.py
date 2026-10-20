import asyncio
import os
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# TFLite Loader (Vercel-safe lightweight runtime)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise RuntimeError("Neither tflite-runtime nor tensorflow.lite is available.")

from llm import analyze_wafer, client as gemini_client


# ============================================================
# APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="WaferGPT",
    description="AI Semiconductor Wafer Defect Intelligence",
    version="3.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS & EPHEMERAL STORAGE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TFLITE_MODEL_PATH = BASE_DIR / "wafer_model_dynamic.tflite"

# Vercel allows writing ONLY inside /tmp
TEMP_DIR = Path("/tmp")
UPLOAD_DIR = TEMP_DIR / "uploads"
RESULT_DIR = TEMP_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
GEMINI_MODEL = "gemini-3.7-flash"


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

if not TFLITE_MODEL_PATH.exists():
    raise FileNotFoundError(f"TFLite model not found at: {TFLITE_MODEL_PATH}")

interpreter = tflite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

tflite_lock = threading.Lock()

print("TFLite model loaded successfully.")
print("Input details:", input_details[0]["shape"])
print("Output details:", output_details[0]["shape"])


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_wafer(image: Image.Image) -> np.ndarray:
    """
    WM-811K preprocessing:
    Grayscale -> 3-level wafer quantization (0, 127, 254) -> resize (224, 224) -> RGB -> float32 / 255.
    """
    image = image.convert("L")
    image_np = np.asarray(image, dtype=np.uint8)

    min_value = int(image_np.min())
    max_value = int(image_np.max())

    if max_value > min_value:
        unique_values = np.unique(image_np)
        has_wafer_levels = (
            np.any(np.abs(unique_values.astype(np.int16) - 0) <= 5)
            and np.any(np.abs(unique_values.astype(np.int16) - 127) <= 10)
            and np.any(np.abs(unique_values.astype(np.int16) - 254) <= 5)
        )

        if not has_wafer_levels:
            image_float = image_np.astype(np.float32)
            image_float = ((image_float - min_value) / (max_value - min_value)) * 254.0
            image_np = image_float.astype(np.uint8)

    image_int = image_np.astype(np.int16)
    distances = np.stack(
        [
            np.abs(image_int - 0),
            np.abs(image_int - 127),
            np.abs(image_int - 254),
        ],
        axis=-1,
    )

    nearest_level = np.argmin(distances, axis=-1)

    wafer_map = np.zeros_like(image_np, dtype=np.uint8)
    wafer_map[nearest_level == 0] = 0
    wafer_map[nearest_level == 1] = 127
    wafer_map[nearest_level == 2] = 254

    wafer_map = cv2.resize(
        wafer_map,
        (224, 224),
        interpolation=cv2.INTER_NEAREST,
    )

    wafer_map = cv2.cvtColor(wafer_map, cv2.COLOR_GRAY2RGB)
    wafer_map = wafer_map.astype(np.float32) / 255.0

    return np.expand_dims(wafer_map, axis=0)


# ============================================================
# TFLITE INFERENCE
# ============================================================

def predict_tflite(processed_image: np.ndarray) -> np.ndarray:
    input_tensor = np.asarray(processed_image, dtype=input_details[0]["dtype"])

    with tflite_lock:
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()
        result = interpreter.get_tensor(output_details[0]["index"])

    result = np.asarray(result, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"Unexpected TFLite output shape: {result.shape}")

    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    # Softmax normalization if model outputs raw logits
    if np.any(result < 0) or not np.isclose(np.sum(result[0]), 1.0, atol=1e-2):
        exp_res = np.exp(result[0] - np.max(result[0]))
        return exp_res / np.sum(exp_res)

    return result[0]


# ============================================================
# DEFECT SALIENCE HEATMAP GENERATION (LIGHTWEIGHT / TFLITE-COMPATIBLE)
# ============================================================

def generate_defect_heatmap(
    image: np.ndarray,
    predicted_class: str,
    output_path: Path,
) -> bool:
    """
    Computes visual defect salience directly without requiring Keras backpropagation.
    Applies Gaussian density distribution over anomalous pixel regions and overlays COLORMAP_JET.
    """
    try:
        wafer_slice = image[0]  # shape (224, 224, 3), range [0, 1]
        original_bgr = (wafer_slice * 255.0).clip(0, 255).astype(np.uint8)
        original_bgr = cv2.cvtColor(original_bgr, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

        # In quantized WM-811K wafers:
        # 0 = background, 127 = normal dies, 254 = defective dies
        if predicted_class.lower() == "none":
            # For defect-free wafers, display nominal uniform low-intensity activation
            heatmap = np.zeros_like(gray, dtype=np.float32)
        else:
            defect_mask = np.where(gray >= 200, 1.0, 0.0).astype(np.float32)
            if np.sum(defect_mask) == 0:
                # If non-quantized input, flag regions differing from typical die intensity (127)
                defect_mask = np.where((gray > 180) | ((gray > 0) & (gray < 70)), 1.0, 0.0).astype(np.float32)

            # Apply spatial diffusion kernel to represent receptive field activation
            heatmap = cv2.GaussianBlur(defect_mask, (31, 31), sigmaX=9, sigmaY=9)
            max_v = np.max(heatmap)
            if max_v > 0:
                heatmap = heatmap / max_v

        heatmap_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(original_bgr, 0.6, colored_heatmap, 0.4, 0)
        return bool(cv2.imwrite(str(output_path), overlay))

    except Exception as exc:
        print("Heatmap generation error:", exc)
        return False


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "name": "WaferGPT",
        "status": "online",
        "version": "3.2.0",
        "engine": "TFLite Dynamic + Gemini",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "tflite_model_loaded": interpreter is not None,
        "gemini": gemini_client is not None,
    }


@app.get("/uploads/{filename}")
async def get_uploaded_image(filename: str):
    safe_name = Path(filename).name
    file_path = UPLOAD_DIR / safe_name
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)


@app.get("/gradcam/{filename}")
async def get_gradcam(filename: str):
    safe_name = Path(filename).name
    file_path = RESULT_DIR / safe_name
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Heatmap not found")
    return FileResponse(file_
