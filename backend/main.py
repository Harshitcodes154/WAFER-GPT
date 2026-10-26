"""
WAFER-GPT FastAPI backend
Pipeline: upload -> TFLite prediction -> probabilities/decision -> Grad-CAM -> AI analysis/chat.

Put this file in backend/main.py with:
  backend/wafer_model_dynamic.tflite
  backend/wafer_model.keras   (required for true Grad-CAM)

Render Root Directory: backend
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import io
import os
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wafer-gpt")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TFLITE_PATH = Path(os.getenv(
    "TFLITE_MODEL_PATH", str(BASE_DIR / "wafer_model_dynamic.tflite")
))
KERAS_PATH = Path(os.getenv(
    "KERAS_MODEL_PATH", str(BASE_DIR / "wafer_model.keras")
))
if not KERAS_PATH.exists() and "KERAS_MODEL_PATH" not in os.environ:
    for name in ("model.keras", "wafer_model.h5", "model.h5",
                 "best_model.keras", "best_model.h5"):
        candidate = BASE_DIR / name
        if candidate.exists():
            KERAS_PATH = candidate
            break

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(4 * 1024 * 1024)))

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
    "Near-full", "Random", "Scratch", "none"
]

app = FastAPI(title="WAFER-GPT API", version="2.1.0")
origins = [x.strip() for x in os.getenv("FRONTEND_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if origins == ["*"] else origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Model loading ----------------
interpreter = None
input_details = None
output_details = None
keras_model = None
keras_api = None
keras_load_error = None
tflite_load_error = None


def load_tflite():
    """Load TFLite and resolve dynamic dimensions to a batch of one image."""
    global interpreter, input_details, output_details, tflite_load_error
    if not TFLITE_PATH.is_file():
        tflite_load_error = f"TFLite model not found: {TFLITE_PATH}"
        log.error(tflite_load_error)
        return

    try:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter

        candidate = Interpreter(model_path=str(TFLITE_PATH))
        candidate.allocate_tensors()
        details = candidate.get_input_details()
        if not details:
            raise RuntimeError("TFLite model has no input tensor.")

        first = details[0]
        shape = [int(v) for v in first["shape"]]
        signature = [int(v) for v in first.get("shape_signature", first["shape"])]

        if len(shape) == 4:
            # Preserve NCHW if channel dimension is the second axis; otherwise NHWC.
            target = ([1, 3, IMAGE_SIZE, IMAGE_SIZE]
                      if len(signature) == 4 and signature[1] == 3 and signature[-1] != 3
                      else [1, IMAGE_SIZE, IMAGE_SIZE, 3])
        elif len(shape) == 3:
            target = [IMAGE_SIZE, IMAGE_SIZE, 3]
        else:
            raise RuntimeError(f"Unsupported TFLite input shape: {shape}")

        if shape != target or any(v <= 0 for v in signature):
            candidate.resize_tensor_input(first["index"], target, strict=False)
            candidate.allocate_tensors()

        interpreter = candidate
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        if not output_details:
            raise RuntimeError("TFLite model has no output tensor.")
        tflite_load_error = None
        log.info("Loaded TFLite model: %s", TFLITE_PATH)
        log.info("TFLite input: %s", input_details[0])
        log.info("TFLite output: %s", output_details[0])
    except Exception as exc:
        interpreter = None
        input_details = None
        output_details = None
        tflite_load_error = f"{type(exc).__name__}: {exc}"
        log.exception("Could not load TFLite model")


def load_keras_for_gradcam():
    """Load the original Keras model separately; failure must not disable TFLite."""
    global keras_model, keras_api, keras_load_error

    if not KERAS_PATH.is_file():
        keras_load_error = f"Keras model not found: {KERAS_PATH}"
        log.warning(keras_load_error)
        return

    errors = []
    # TensorFlow 2.17 uses Keras 3 by default. Try the standalone Keras loader first.
    try:
        import keras
        loaded = keras.models.load_model(str(KERAS_PATH), compile=False)
        keras_model, keras_api, keras_load_error = loaded, keras, None
        log.info("Loaded Keras model with keras %s from %s",
                 getattr(keras, "__version__", "unknown"), KERAS_PATH)
        return
    except Exception as exc:
        errors.append(f"keras.models.load_model: {type(exc).__name__}: {exc}")
        log.exception("Standalone Keras model load failed")

    try:
        import tensorflow as tf
        loaded = tf.keras.models.load_model(str(KERAS_PATH), compile=False)
        keras_model, keras_api, keras_load_error = loaded, tf.keras, None
        log.info("Loaded Keras model with tf.keras / TF %s from %s",
                 tf.__version__, KERAS_PATH)
        return
    except Exception as exc:
        errors.append(f"tf.keras.models.load_model: {type(exc).__name__}: {exc}")
        log.exception("tf.keras model load failed")

    keras_model = None
    keras_api = None
    keras_load_error = " | ".join(errors)
    log.error("Grad-CAM model unavailable: %s", keras_load_error)


load_tflite()
load_keras_for_gradcam()


# ---------------- Image preprocessing / inference ----------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Match the project's known preprocessing: grayscale -> RGB -> 224x224 -> [0,1]."""
    image = image.convert("L").convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def prepare_for_tflite(batch: np.ndarray) -> np.ndarray:
    if interpreter is None or not input_details:
        raise RuntimeError(tflite_load_error or "TFLite model is not loaded.")

    detail = input_details[0]
    shape = [int(v) for v in detail["shape"]]
    dtype = detail["dtype"]
    arr = batch

    if len(shape) == 4:
        nchw = shape[1] == 3 and shape[-1] != 3
        height, width = (shape[2], shape[3]) if nchw else (shape[1], shape[2])
        if arr.shape[1:3] != (height, width):
            img = Image.fromarray(np.uint8(np.clip(arr[0] * 255.0, 0, 255)))
            img = img.resize((width, height), Image.Resampling.BILINEAR)
            arr = np.expand_dims(np.asarray(img, dtype=np.float32) / 255.0, 0)
        if nchw:
            arr = np.transpose(arr, (0, 3, 1, 2))
    elif len(shape) == 3:
        height, width = shape[0], shape[1]
        if arr.shape[1:3] != (height, width):
            img = Image.fromarray(np.uint8(np.clip(arr[0] * 255.0, 0, 255)))
            img = img.resize((width, height), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0

    if np.issubdtype(dtype, np.integer):
        scale, zero_point = detail.get("quantization", (0.0, 0))
        if scale and scale > 0:
            arr = np.round(arr / scale + zero_point)
        else:
            arr = np.round(arr * 255.0)
        limits = np.iinfo(dtype)
        return np.clip(arr, limits.min, limits.max).astype(dtype)
    return arr.astype(dtype)


def normalize_probabilities(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Model returned NaN or infinite outputs.")

    # Convert logits to softmax only if output does not already look like probabilities.
    if np.any(values < 0) or not np.isclose(np.sum(values), 1.0, atol=0.03):
        values = values - np.max(values)
        exp_values = np.exp(values)
        values = exp_values / max(float(np.sum(exp_values)), 1e-12)
    else:
        values = values / max(float(np.sum(values)), 1e-12)

    if len(values) != len(CLASS_NAMES):
        raise RuntimeError(
            f"Model returned {len(values)} values but {len(CLASS_NAMES)} class labels "
            "are configured. Check CLASS_NAMES against the trained model."
        )
    return values.astype(np.float32)


def predict(batch: np.ndarray) -> np.ndarray:
    if interpreter is None:
        raise RuntimeError(tflite_load_error or "TFLite model is not loaded.")
    model_input = prepare_for_tflite(batch)
    interpreter.set_tensor(input_details[0]["index"], model_input)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_details[0]["index"])
    return normalize_probabilities(raw)


def make_decision(prediction: str, confidence: float) -> Dict[str, Any]:
    # These are heuristic UI thresholds, not validated manufacturing safety thresholds.
    if confidence >= 0.80:
        status = "High confidence"
        recommendation = f"The model strongly predicts the {prediction} pattern."
    elif confidence >= 0.55:
        status = "Moderate confidence"
        recommendation = f"The likely pattern is {prediction}; review the wafer image and heatmap."
    else:
        status = "Low confidence — manual review recommended"
        recommendation = "The model is uncertain. Do not rely on this prediction alone."
    return {
        "status": status,
        "recommendation": recommendation,
        "confidence_threshold_note": "Heuristic thresholds; validate against your dataset."
    }


# ---------------- Grad-CAM ----------------
def generate_gradcam(batch: np.ndarray, output_filename: str) -> Optional[str]:
    """Generate a true Grad-CAM overlay from the original Keras model, if loadable."""
    if keras_model is None:
        log.warning("Grad-CAM skipped: %s", keras_load_error)
        return None

    try:
        import tensorflow as tf
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        api = keras_api
        conv_layers = [
            layer for layer in keras_model.layers
            if isinstance(layer, api.layers.Conv2D)
        ]
        if not conv_layers:
            log.warning("Grad-CAM skipped: no top-level Conv2D layer was found.")
            return None

        last_conv = conv_layers[-1]
        grad_model = api.Model(
            inputs=keras_model.inputs,
            outputs=[last_conv.output, keras_model.output]
        )

        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(
                tf.convert_to_tensor(batch, dtype=tf.float32), training=False
            )
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            if isinstance(predictions, dict):
                predictions = next(iter(predictions.values()))
            class_index = tf.argmax(predictions[0])
            score = predictions[:, class_index]

        gradients = tape.gradient(score, conv_output)
        if gradients is None:
            log.warning("Grad-CAM skipped because gradients were unavailable.")
            return None

        weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
        feature_map = conv_output[0]
        heatmap = tf.reduce_sum(feature_map * weights, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()

        heat_img = Image.fromarray(np.uint8(np.clip(heatmap * 255, 0, 255)))
        heat_img = heat_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        heat = np.asarray(heat_img, dtype=np.float32) / 255.0
        base = np.uint8(np.clip(batch[0] * 255.0, 0, 255))

        target = RESULT_DIR / Path(output_filename).name
        fig = plt.figure(figsize=(5, 5))
        try:
            plt.imshow(base)
            plt.imshow(heat, cmap="jet", alpha=0.40, vmin=0, vmax=1)
            plt.axis("off")
            fig.tight_layout(pad=0)
            fig.savefig(target, bbox_inches="tight", pad_inches=0, dpi=140)
        finally:
            plt.close(fig)
        return f"/gradcam/{target.name}"
    except Exception:
        log.exception("Grad-CAM generation failed")
        return None


# ---------------- NVIDIA-compatible assistant ----------------
def ask_nvidia(messages):
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set NVIDIA_API_KEY in Render environment variables.")

    from openai import OpenAI
    client = OpenAI(
        base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=api_key,
        timeout=45.0,
    )
    model_name = os.getenv("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=700,
    )
    return response.choices[0].message.content or "I could not generate a response."


def make_initial_analysis(prediction, confidence, probabilities, decision):
    if not (os.getenv("NVIDIA_API_KEY") or os.getenv("GEMINI_API_KEY")):
        return (
            f"Predicted pattern: {prediction}. Confidence: {confidence:.1%}. "
            f"Decision: {decision['status']}. Configure NVIDIA_API_KEY to enable AI analysis/chat."
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You are WaferGPT, an assistant for semiconductor wafer-map classification. "
                "Use only the provided model outputs. Do not claim certainty or infer a "
                "manufacturing root cause without evidence. Explain plainly and concisely."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Prediction: {prediction}\nConfidence: {confidence:.1%}\n"
                f"Probabilities: {probabilities}\nDecision: {decision}\n"
                "Give a concise interpretation, limitations, and suggested next checks."
            ),
        },
    ]
    try:
        return ask_nvidia(messages)
    except Exception as exc:
        log.exception("Initial AI analysis failed")
        return (
            f"Prediction: {prediction} ({confidence:.1%}). Decision: {decision['status']}. "
            f"AI analysis could not be generated: {str(exc)[:250]}"
        )


# ---------------- API endpoints ----------------
@app.get("/")
def root():
    return {
        "service": "WAFER-GPT API",
        "status": "running",
        "flow": ["image input", "prediction", "probabilities", "decision", "gradcam", "chat"],
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if interpreter is not None else "model_not_loaded",
        "tflite_model_loaded": interpreter is not None,
        "tflite_model_path": str(TFLITE_PATH),
        "tflite_error": tflite_load_error,
        "keras_model_loaded_for_gradcam": keras_model is not None,
        "keras_model_path": str(KERAS_PATH),
        "keras_version_loader": getattr(keras_api, "__name__", None) if keras_api else None,
        "gradcam_note": "Ready" if keras_model is not None else keras_load_error,
    }


@app.get("/uploads/{filename}")
def get_upload(filename: str):
    target = UPLOAD_DIR / Path(filename).name
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Uploaded image not found.")
    return FileResponse(target, media_type="image/png")


@app.get("/gradcam/{filename}")
def get_gradcam(filename: str):
    target = RESULT_DIR / Path(filename).name
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Grad-CAM image not found.")
    return FileResponse(target, media_type="image/png")


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image is too large. Maximum allowed is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        batch = preprocess_image(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Upload a valid PNG/JPG.")

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
            prediction, confidence, probabilities, decision
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
                else f"True Grad-CAM unavailable: {keras_load_error or 'Could not create heatmap.'}"
            ),
            "llm_analysis": llm_analysis,
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("[%s] Prediction pipeline failed", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction pipeline failed: {str(exc)[:300]}"
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
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    context = (
        f"Current model prediction: {body.prediction or 'Not provided'}\n"
        f"Confidence: {body.confidence if body.confidence is not None else 'Not provided'}\n"
        f"Class probabilities: {body.probabilities or 'Not provided'}\n"
        f"Decision: {body.decision or 'Not provided'}\n"
        f"Previous AI analysis: {body.llm_analysis or 'Not provided'}"
    )
    messages = [{
        "role": "system",
        "content": (
            "You are WaferGPT, a helpful semiconductor wafer-map assistant. Use the provided "
            "prediction context. Explain probabilities and Grad-CAM carefully. Never present "
            "model output as a guaranteed diagnosis or claim the heatmap proves causality. "
            "If no prediction context is available, ask the user to upload an image first.\n\n"
            "Context:\n" + context
        ),
    }]

    for item in (body.history or [])[-8:]:
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}:
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                messages.append({"role": item["role"], "content": content[:2000]})
    messages.append({"role": "user", "content": message})

    try:
        return {"success": True, "reply": ask_nvidia(messages)}
    except Exception as exc:
        log.exception("Chat assistant failed")
        raise HTTPException(
            status_code=502,
            detail=f"Chat assistant unavailable: {str(exc)[:250]}"
        )
