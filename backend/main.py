
"""
WAFER-GPT FastAPI backend — Keras-only version.

Model:
    backend/wafer_model.keras

Render:
    Root Directory: backend
    Start Command:
        uvicorn main:app --host 0.0.0.0 --port $PORT

Pipeline:
    Upload -> Preprocessing -> Keras prediction -> Probabilities
    -> Decision -> Grad-CAM -> AI analysis/chat
"""

import io
import os
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wafer-gpt")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
KERAS_PATH = Path(
    os.getenv("KERAS_MODEL_PATH", str(BASE_DIR / "wafer_model.keras"))
)

UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 64
MAX_UPLOAD_BYTES = int(
    os.getenv("MAX_UPLOAD_BYTES", str(4 * 1024 * 1024))
)

# Must match the class order used to train the final weighted model.
CLASS_NAMES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
    "none",
]

app = FastAPI(
    title="WAFER-GPT API",
    version="3.0.0",
    description="Wafer defect classification using a Keras model.",
)

origins = [
    item.strip()
    for item in os.getenv("FRONTEND_ORIGINS", "*").split(",")
    if item.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if origins == ["*"] else origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# MODEL LOADING
# ============================================================

keras_model = None
keras_load_error = None


def load_keras_model():
    """Load and validate the final Keras model."""

    global keras_model, keras_load_error

    if not KERAS_PATH.is_file():
        keras_load_error = f"Model file not found: {KERAS_PATH}"
        log.error(keras_load_error)
        return

    try:
        keras_model = tf.keras.models.load_model(
            str(KERAS_PATH),
            compile=False,
        )

        input_shape = tuple(keras_model.input_shape[1:])
        output_count = int(keras_model.output_shape[-1])

        if input_shape != (64, 64, 1):
            raise ValueError(
                f"Expected model input (64, 64, 1); got {input_shape}"
            )

        if output_count != len(CLASS_NAMES):
            raise ValueError(
                f"Model outputs {output_count} classes, "
                f"but CLASS_NAMES has {len(CLASS_NAMES)} entries."
            )

        keras_load_error = None

        log.info("Keras model loaded: %s", KERAS_PATH)
        log.info("Model input shape: %s", keras_model.input_shape)
        log.info("Model output shape: %s", keras_model.output_shape)
        log.info("TensorFlow version: %s", tf.__version__)

    except Exception as exc:
        keras_model = None
        keras_load_error = f"{type(exc).__name__}: {exc}"
        log.exception("Could not load Keras model")


load_keras_model()

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Convert an uploaded image to (1, 64, 64, 1).

    IMPORTANT:
    This is a heuristic conversion for external images.
    Verify it against the exact preprocessing used during training.
    WM-811K maps contain categorical values 0, 1, 2; grayscale
    screenshots do not necessarily preserve those values exactly.
    """

    gray = np.asarray(image.convert("L"), dtype=np.uint8)

    # Approximate grayscale-to-category mapping.
    # Adjust these thresholds if your exported map uses different colors.
    categorical = np.where(
        gray < 64,
        0,
        np.where(gray < 192, 1, 2),
    ).astype(np.uint8)

    # Pad to a square while preserving aspect ratio.
    height, width = categorical.shape
    side = max(height, width)

    top = (side - height) // 2
    bottom = side - height - top
    left = (side - width) // 2
    right = side - width - left

    square = np.pad(
        categorical,
        ((top, bottom), (left, right)),
        mode="constant",
        constant_values=0,
    )

    # Nearest-neighbour resize preserves categorical pixel values.
    resized = Image.fromarray(square).resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST,
    )

    arr = np.asarray(resized, dtype=np.float32)

    # Shape: batch x height x width x channels.
    # Values remain 0, 1, 2 — no division by 255.
    return arr[np.newaxis, ..., np.newaxis]


# ============================================================
# PREDICTION
# ============================================================

def normalize_probabilities(raw: np.ndarray) -> np.ndarray:
    """Validate the model's nine output scores."""

    values = np.asarray(raw, dtype=np.float64).reshape(-1)

    if not np.all(np.isfinite(values)):
        raise RuntimeError("Model returned NaN or infinite scores.")

    if len(values) != len(CLASS_NAMES):
        raise RuntimeError(
            f"Model returned {len(values)} scores; "
            f"expected {len(CLASS_NAMES)}."
        )

    # This model is expected to end with a softmax activation.
    if (
        np.any(values < 0)
        or np.any(values > 1)
        or not np.isclose(values.sum(), 1.0, atol=0.03)
    ):
        raise RuntimeError(
            "Output does not look like softmax probabilities. "
            "Verify the model's final activation."
        )

    total = float(values.sum())
    if total <= 0:
        raise RuntimeError("Model returned an invalid probability sum.")

    return (values / total).astype(np.float32)


def predict(batch: np.ndarray) -> np.ndarray:
    """Run inference using the Keras model only."""

    if keras_model is None:
        raise RuntimeError(
            keras_load_error or "Keras model is not loaded."
        )

    expected_shape = tuple(keras_model.input_shape[1:])
    if tuple(batch.shape[1:]) != expected_shape:
        raise ValueError(
            f"Input shape {batch.shape[1:]} does not match "
            f"model input {expected_shape}."
        )

    raw = keras_model.predict(batch, verbose=0)[0]
    return normalize_probabilities(raw)


def make_decision(
    prediction: str,
    confidence: float,
) -> Dict[str, Any]:
    """UI guidance only; these thresholds are not validated safety limits."""

    if confidence >= 0.80:
        status = "High confidence"
        recommendation = (
            f"The model strongly predicts the {prediction} pattern."
        )
    elif confidence >= 0.55:
        status = "Moderate confidence"
        recommendation = (
            f"The likely pattern is {prediction}; review the image and heatmap."
        )
    else:
        status = "Low confidence — manual review recommended"
        recommendation = (
            "The model is uncertain. Do not rely on this prediction alone."
        )

    return {
        "status": status,
        "recommendation": recommendation,
        "confidence_threshold_note": (
            "Heuristic thresholds; validate against labelled test data."
        ),
    }


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    batch: np.ndarray,
    output_filename: str,
) -> Optional[str]:
    """Create a Grad-CAM overlay using the Keras model."""

    if keras_model is None:
        log.warning("Grad-CAM unavailable: %s", keras_load_error)
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        conv_layers = [
            layer
            for layer in keras_model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]

        if not conv_layers:
            log.warning(
                "Grad-CAM unavailable: no top-level Conv2D layer found."
            )
            return None

        last_conv = conv_layers[-1]

        grad_model = tf.keras.Model(
            inputs=keras_model.inputs,
            outputs=[last_conv.output, keras_model.output],
        )

        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(
                tf.convert_to_tensor(batch, dtype=tf.float32),
                training=False,
            )

            class_index = tf.argmax(predictions[0])
            score = predictions[:, class_index]

        gradients = tape.gradient(score, conv_output)

        if gradients is None:
            log.warning("Grad-CAM gradients are unavailable.")
            return None

        weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
        feature_map = conv_output[0]

        heatmap = tf.reduce_sum(feature_map * weights, axis=-1)
        heatmap = tf.maximum(heatmap, 0)

        max_heat = tf.reduce_max(heatmap)
        heatmap = heatmap / (max_heat + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()

        heat_img = Image.fromarray(
            np.uint8(np.clip(heatmap * 255, 0, 255))
        )
        heat_img = heat_img.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )
        heat = np.asarray(heat_img, dtype=np.float32) / 255.0

        # Display categorical values as grayscale RGB.
        base_gray = np.uint8(
            np.clip(batch[0, :, :, 0] * 127.5, 0, 255)
        )
        base_rgb = np.repeat(
            base_gray[..., np.newaxis],
            3,
            axis=-1,
        )

        target = RESULT_DIR / Path(output_filename).name

        fig = plt.figure(figsize=(5, 5))
        try:
            plt.imshow(base_rgb)
            plt.imshow(
                heat,
                cmap="jet",
                alpha=0.40,
                vmin=0,
                vmax=1,
            )
            plt.axis("off")
            fig.tight_layout(pad=0)
            fig.savefig(
                target,
                bbox_inches="tight",
                pad_inches=0,
                dpi=140,
            )
        finally:
            plt.close(fig)

        return f"/gradcam/{target.name}"

    except Exception:
        log.exception("Grad-CAM generation failed")
        return None


# ============================================================
# AI ANALYSIS AND CHAT
# ============================================================

def ask_nvidia(messages):
    api_key = (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    )

    if not api_key:
        raise RuntimeError(
            "Set NVIDIA_API_KEY in Render environment variables."
        )

    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv(
            "NVIDIA_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        ),
        api_key=api_key,
        timeout=45.0,
    )

    model_name = os.getenv(
        "NVIDIA_MODEL",
        "meta/llama-3.2-11b-vision-instruct",
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=700,
    )

    return (
        response.choices[0].message.content
        or "I could not generate a response."
    )


def make_initial_analysis(
    prediction,
    confidence,
    probabilities,
    decision,
):
    if not (
        os.getenv("NVIDIA_API_KEY")
        or os.getenv("GEMINI_API_KEY")
    ):
        return (
            f"Predicted pattern: {prediction}. "
            f"Confidence: {confidence:.1%}. "
            f"Decision: {decision['status']}. "
            "Configure NVIDIA_API_KEY to enable AI analysis/chat."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are WaferGPT, an assistant for semiconductor wafer-map "
                "classification. Use only the provided model outputs. Do not "
                "claim certainty or infer a manufacturing root cause without "
                "evidence. Explain plainly and concisely."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Prediction: {prediction}\n"
                f"Confidence: {confidence:.1%}\n"
                f"Probabilities: {probabilities}\n"
                f"Decision: {decision}\n"
                "Give a concise interpretation, limitations, and suggested checks."
            ),
        },
    ]

    try:
        return ask_nvidia(messages)
    except Exception as exc:
        log.exception("Initial AI analysis failed")
        return (
            f"Prediction: {prediction} ({confidence:.1%}). "
            f"Decision: {decision['status']}. "
            f"AI analysis could not be generated: {str(exc)[:250]}"
        )


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "service": "WAFER-GPT API",
        "status": "running",
        "model_type": "Keras",
        "flow": [
            "image input",
            "prediction",
            "probabilities",
            "decision",
            "gradcam",
            "chat",
        ],
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    loaded = keras_model is not None

    return {
        "status": "ok" if loaded else "model_not_loaded",
        "keras_model_loaded": loaded,
        "keras_model_path": str(KERAS_PATH),
        "keras_load_error": keras_load_error,
        "input_shape": (
            list(keras_model.input_shape) if loaded else None
        ),
        "output_shape": (
            list(keras_model.output_shape) if loaded else None
        ),
        "classes": CLASS_NAMES,
    }


@app.get("/uploads/{filename}")
def get_upload(filename: str):
    target = UPLOAD_DIR / Path(filename).name

    if not target.is_file():
        raise HTTPException(
            status_code=404,
            detail="Uploaded image not found.",
        )

    return FileResponse(target, media_type="image/png")


@app.get("/gradcam/{filename}")
def get_gradcam(filename: str):
    target = RESULT_DIR / Path(filename).name

    if not target.is_file():
        raise HTTPException(
            status_code=404,
            detail="Grad-CAM image not found.",
        )

    return FileResponse(target, media_type="image/png")


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Please upload an image file.",
        )

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "Image is too large. Maximum allowed is "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            ),
        )

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        batch = preprocess_image(image)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read image. Upload a valid PNG/JPG.",
        )

    request_id = uuid.uuid4().hex[:12]
    saved_name = f"{request_id}.png"
    image.save(UPLOAD_DIR / saved_name, format="PNG")

    try:
        probs = predict(batch)

        idx = int(np.argmax(probs))
        prediction = CLASS_NAMES[idx]
        confidence = float(probs[idx])

        probabilities = {
            CLASS_NAMES[i]: round(float(probs[i]), 6)
            for i in range(len(CLASS_NAMES))
        }

        decision = make_decision(prediction, confidence)

        gradcam_name = f"{request_id}_gradcam.png"
        gradcam_url = generate_gradcam(batch, gradcam_name)

        llm_analysis = make_initial_analysis(
            prediction,
            confidence,
            probabilities,
            decision,
        )

        return {
            "success": True,
            "request_id": request_id,
            "filename": saved_name,
            "image_url": f"/uploads/{saved_name}",
            "prediction": prediction,
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 2),
            "probabilities": probabilities,
            "decision": decision,
            "gradcam_url": gradcam_url,
            "gradcam_available": gradcam_url is not None,
            "gradcam_note": (
                "Grad-CAM overlay generated."
                if gradcam_url
                else "Grad-CAM could not be generated; see backend logs."
            ),
            "llm_analysis": llm_analysis,
        }

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("[%s] Prediction pipeline failed", request_id)

        raise HTTPException(
            status_code=500,
            detail=f"Prediction pipeline failed: {str(exc)[:300]}",
        )


class ChatRequest(BaseModel):
    message: str
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    decision: Optional[Dict[str, Any]] = None
    llm_analysis: Optional[str] = None
    history: Optional[list] = None


@app.post("/chat")
def chat_endpoint(body: ChatRequest):
    message = body.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    context = (
        f"Current model prediction: {body.prediction or 'Not provided'}\n"
        f"Confidence: "
        f"{body.confidence if body.confidence is not None else 'Not provided'}\n"
        f"Class probabilities: {body.probabilities or 'Not provided'}\n"
        f"Decision: {body.decision or 'Not provided'}\n"
        f"Previous AI analysis: {body.llm_analysis or 'Not provided'}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are WaferGPT, a helpful semiconductor wafer-map assistant. "
                "Use the provided prediction context. Explain probabilities and "
                "Grad-CAM carefully. Never present model output as a guaranteed "
                "diagnosis or claim the heatmap proves causality. If no prediction "
                "context is available, ask the user to upload an image first.\n\n"
                "Context:\n" + context
            ),
        }
    ]

    for item in (body.history or [])[-8:]:
        if (
            isinstance(item, dict)
            and item.get("role") in {"user", "assistant"}
        ):
            content = item.get("content")

            if isinstance(content, str) and content.strip():
                messages.append({
                    "role": item["role"],
                    "content": content[:2000],
                })

    messages.append({
        "role": "user",
        "content": message,
    })

    try:
        return {
            "success": True,
            "reply": ask_nvidia(messages),
        }
    except Exception as exc:
        log.exception("Chat assistant failed")

        raise HTTPException(
            status_code=502,
            detail=f"Chat assistant unavailable: {str(exc)[:250]}",
        )
