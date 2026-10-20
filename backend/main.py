
import asyncio
import base64
import os
import threading
import uuid
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI


# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="WaferGPT",
    description="AI Semiconductor Wafer Defect Intelligence",
    version="3.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# NVIDIA API CONFIGURATION
# ============================================================

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

NVIDIA_MODEL = os.getenv(
    "NVIDIA_MODEL",
    "meta/llama-3.2-11b-vision-instruct",
)

nvidia_client = (
    OpenAI(
        api_key=NVIDIA_API_KEY,
        base_url="https://integrate.api.nvidia.com/v1",
        timeout=45.0,
        max_retries=1,
    )
    if NVIDIA_API_KEY
    else None
)


# ============================================================
# PATHS AND TEMPORARY STORAGE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

TFLITE_MODEL_PATH = BASE_DIR / "wafer_model_dynamic.tflite"

TEMP_DIR = Path("/tmp")
UPLOAD_DIR = TEMP_DIR / "uploads"
RESULT_DIR = TEMP_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Vercel request bodies have a 4.5 MB limit.
# Keep the uploaded image below that limit.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


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

if not TFLITE_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"TFLite model not found: {TFLITE_MODEL_PATH}"
    )

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError as exc:
            raise RuntimeError(
                "No compatible TFLite interpreter installed. "
                "Install ai-edge-litert."
            ) from exc


interpreter = tflite.Interpreter(
    model_path=str(TFLITE_MODEL_PATH)
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

tflite_lock = threading.Lock()

print("WaferGPT TFLite model loaded.")
print("Model input:", input_details[0]["shape"])
print("Model output:", output_details[0]["shape"])


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_wafer(image: Image.Image) -> np.ndarray:
    """
    Grayscale -> wafer-level quantization ->
    224x224 resize -> RGB -> float32 / 255.
    """

    image = image.convert("L")
    image_np = np.asarray(image, dtype=np.uint8)

    min_value = int(image_np.min())
    max_value = int(image_np.max())

    if max_value > min_value:
        unique_values = np.unique(image_np.astype(np.int16))

        has_wafer_levels = (
            np.any(np.abs(unique_values - 0) <= 5)
            and np.any(np.abs(unique_values - 127) <= 10)
            and np.any(np.abs(unique_values - 254) <= 5)
        )

        if not has_wafer_levels:
            image_float = image_np.astype(np.float32)
            image_float = (
                (image_float - min_value)
                / (max_value - min_value)
            ) * 254.0

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

    wafer_map = cv2.cvtColor(
        wafer_map,
        cv2.COLOR_GRAY2RGB,
    )

    wafer_map = wafer_map.astype(np.float32) / 255.0

    return np.expand_dims(wafer_map, axis=0)


# ============================================================
# TFLITE INFERENCE
# ============================================================

def predict_tflite(
    processed_image: np.ndarray,
) -> np.ndarray:

    input_tensor = np.asarray(
        processed_image,
        dtype=input_details[0]["dtype"],
    )

    with tflite_lock:
        interpreter.set_tensor(
            input_details[0]["index"],
            input_tensor,
        )
        interpreter.invoke()

        result = interpreter.get_tensor(
            output_details[0]["index"]
        )

    result = np.asarray(result, dtype=np.float32)

    if (
        result.ndim != 2
        or result.shape[0] != 1
        or result.shape[1] != len(CLASS_NAMES)
    ):
        raise ValueError(
            f"Unexpected TFLite output shape: {result.shape}"
        )

    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Convert logits to probabilities when needed.
    if (
        np.any(result < 0)
        or not np.isclose(
            np.sum(result[0]),
            1.0,
            atol=1e-2,
        )
    ):
        logits = result[0] - np.max(result[0])
        exp_result = np.exp(logits)
        total = np.sum(exp_result)

        if not np.isfinite(total) or total <= 0:
            raise ValueError("Model returned invalid scores.")

        return exp_result / total

    # Normalize small rounding differences.
    total = np.sum(result[0])

    if total <= 0:
        raise ValueError("Model returned invalid probabilities.")

    return result[0] / total


# ============================================================
# DEFECT SALIENCE HEATMAP
# ============================================================

def generate_defect_heatmap(
    image: np.ndarray,
    predicted_class: str,
    output_path: Path,
) -> bool:
    """
    Creates a visual defect-salience overlay.
    This is a heuristic heatmap, not a true Grad-CAM.
    """

    try:
        wafer_slice = image[0]

        original_bgr = (
            wafer_slice * 255.0
        ).clip(0, 255).astype(np.uint8)

        original_bgr = cv2.cvtColor(
            original_bgr,
            cv2.COLOR_RGB2BGR,
        )

        gray = cv2.cvtColor(
            original_bgr,
            cv2.COLOR_BGR2GRAY,
        )

        if predicted_class.lower() == "none":
            heatmap = np.zeros_like(
                gray,
                dtype=np.float32,
            )
        else:
            defect_mask = (
                gray >= 200
            ).astype(np.float32)

            if np.sum(defect_mask) == 0:
                defect_mask = (
                    (gray > 180)
                    | ((gray > 0) & (gray < 70))
                ).astype(np.float32)

            heatmap = cv2.GaussianBlur(
                defect_mask,
                (31, 31),
                sigmaX=9,
                sigmaY=9,
            )

            max_value = float(np.max(heatmap))

            if max_value > 0:
                heatmap /= max_value

        heatmap_uint8 = (
            heatmap * 255
        ).clip(0, 255).astype(np.uint8)

        colored_heatmap = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET,
        )

        overlay = cv2.addWeighted(
            original_bgr,
            0.6,
            colored_heatmap,
            0.4,
            0,
        )

        return bool(
            cv2.imwrite(str(output_path), overlay)
        )

    except Exception as exc:
        print("Heatmap generation error:", exc)
        return False


# ============================================================
# NVIDIA VISION ANALYSIS
# ============================================================

def analyze_wafer_with_nvidia(
    predicted_class: str,
    confidence: float,
    probabilities: dict,
    image_path: str,
) -> str:

    if nvidia_client is None:
        return (
            "NVIDIA analysis unavailable. "
            "Configure NVIDIA_API_KEY in Render Environment Variables."
        )

    prompt = f"""
You are WaferGPT, a semiconductor wafer defect
analysis assistant.

CNN prediction: {predicted_class}
CNN confidence: {confidence:.2f}%
Class probabilities: {probabilities}

Analyze the provided wafer map using its visual
appearance and the CNN prediction.

Do not claim a confirmed manufacturing diagnosis.
Describe possible causes as hypotheses, not facts.
Distinguish the AI prediction from engineering verification.

Use these sections:

DEFECT ANALYSIS
VISUAL EVIDENCE
WHY THIS PREDICTION
POSSIBLE ROOT CAUSES
RECOMMENDED INSPECTION
SEVERITY
ENGINEER SUMMARY
"""

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            buffer = BytesIO()
            image.save(buffer, format="PNG")

            image_b64 = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

        response = nvidia_client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    "data:image/png;base64,"
                                    + image_b64
                                )
                            },
                        },
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=1200,
        )

        answer = response.choices[0].message.content

        if isinstance(answer, str) and answer.strip():
            return answer

        return "NVIDIA returned an empty analysis."

    except Exception as exc:
        print("NVIDIA analysis error:", str(exc))
        return (
            "CNN prediction completed, but NVIDIA analysis "
            "is temporarily unavailable. Check server logs "
            "and NVIDIA model configuration."
        )


# ============================================================
# ROOT AND HEALTH
# ============================================================

@app.get("/")
async def root():
    # Render hosts the API only; the frontend remains on Vercel.
    return {
        "name": "WaferGPT API",
        "status": "online",
        "version": "3.3.0",
        "engine": "TFLite + NVIDIA NIM",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "tflite_model_loaded": interpreter is not None,
        "nvidia_api_configured": nvidia_client is not None,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a PNG, JPG, JPEG or WEBP image.",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 4 MB.",
        )

    upload_path = UPLOAD_DIR / (
        f"{uuid.uuid4().hex}{extension}"
    )

    upload_path.write_bytes(content)

    try:
        # Validate and decode the uploaded image.
        with Image.open(upload_path) as source:
            source.load()
            original_image = source.copy()

        processed_image = preprocess_wafer(original_image)

        scores = await asyncio.to_thread(
            predict_tflite,
            processed_image,
        )

        index = int(np.argmax(scores))
        predicted_class = CLASS_NAMES[index]
        confidence = float(scores[index]) * 100.0

        probabilities = {
            name: float(score)
            for name, score in zip(CLASS_NAMES, scores)
        }

        heatmap_path = RESULT_DIR / (
            f"{upload_path.stem}_heatmap.png"
        )

        heatmap_created = await asyncio.to_thread(
            generate_defect_heatmap,
            processed_image,
            predicted_class,
            heatmap_path,
        )

        # NVIDIA explanations are optional; the CNN result
        # can still be returned if the LLM request fails.
        llm_analysis = await asyncio.to_thread(
            analyze_wafer_with_nvidia,
            predicted_class,
            confidence,
            probabilities,
            str(upload_path),
        )

        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "llm_analysis": llm_analysis,
            "gradcam_url": (
                f"/gradcam/{heatmap_path.name}"
                if heatmap_created
                else None
            ),
        }

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError) as exc:
        print("Image processing error:", exc)

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    except Exception as exc:
        print("Prediction error:", exc)

        raise HTTPException(
            status_code=500,
            detail=(
                "Wafer prediction failed. "
                "Check Render service logs."
            ),
        )

    finally:
        await file.close()


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
async def chat(payload: dict):

    if nvidia_client is None:
        raise HTTPException(
            status_code=503,
            detail="NVIDIA_API_KEY is not configured.",
        )

    message = str(payload.get("message", "")).strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    prediction = str(
        payload.get("prediction", "Unknown")
    )

    confidence = payload.get("confidence", "Unknown")
    probabilities = payload.get("probabilities", {})
    previous_analysis = str(
        payload.get("llm_analysis", "")
    )

    prompt = f"""
You are WaferGPT, an AI semiconductor wafer
engineering assistant.

Current prediction: {prediction}
Confidence: {confidence}%
Class probabilities: {probabilities}
Previous analysis: {previous_analysis}

User question:
{message}

Give a clear, useful answer. Explain uncertainty.
Do not present possible manufacturing root causes
as confirmed facts.
"""

    try:
        response = await asyncio.to_thread(
            lambda: nvidia_client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.2,
                max_tokens=1000,
            )
        )

        answer = response.choices[0].message.content

        if not isinstance(answer, str) or not answer.strip():
            raise HTTPException(
                status_code=502,
                detail="NVIDIA returned an empty response.",
            )

        return {"response": answer}

    except HTTPException:
        raise

    except Exception as exc:
        print("NVIDIA chat error:", str(exc))

        raise HTTPException(
            status_code=502,
            detail="NVIDIA chat request failed.",
        )


# ============================================================
# UPLOAD AND HEATMAP FILE ROUTES
# ============================================================

@app.get("/uploads/{filename}")
async def get_uploaded_image(filename: str):

    safe_name = Path(filename).name
    file_path = UPLOAD_DIR / safe_name

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Image not found.",
        )

    return FileResponse(file_path)


@app.get("/gradcam/{filename}")
async def get_gradcam(filename: str):

    safe_name = Path(filename).name
    file_path = RESULT_DIR / safe_name

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Heatmap not found.",
        )

    return FileResponse(file_path)