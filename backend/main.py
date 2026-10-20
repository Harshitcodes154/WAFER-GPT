"""
WAFER-GPT FastAPI backend
Flow: image upload -> TFLite prediction -> class probabilities -> decision
      -> Grad-CAM (when the original Keras model is available) -> chat assistant.

Expected files:
  backend/wafer_model_dynamic.tflite   (required; dynamic tensor dimensions are resolved at startup)
  backend/wafer_model.keras           (optional; needed for true Grad-CAM)

Render start command:
  uvicorn main:app --host 0.0.0.0 --port $PORT
Run this file from the backend directory.
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

TFLITE_PATH = Path(os.getenv("TFLITE_MODEL_PATH", str(BASE_DIR / "wafer_model_dynamic.tflite")))
KERAS_PATH = Path(os.getenv("KERAS_MODEL_PATH", str(BASE_DIR / "wafer_model.keras")))
# If no explicit KERAS_MODEL_PATH is set, auto-detect common Keras/H5 filenames.
if "KERAS_MODEL_PATH" not in os.environ and not KERAS_PATH.exists():
    for candidate_name in ("model.keras", "wafer_model.h5", "model.h5", "best_model.keras", "best_model.h5"):
        candidate = BASE_DIR / candidate_name
        if candidate.exists():
            KERAS_PATH = candidate
            break
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(4 * 1024 * 1024)))

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
    "Near-full", "Random", "Scratch", "none"
]

app = FastAPI(title="WAFER-GPT API", version="2.0.0")

allowed_origins = [
    x.strip() for x in os.getenv("FRONTEND_ORIGINS", "*").split(",") if x.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allowed_origins == ["*"] else allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Model loading --------
interpreter = None
input_details = None
output_details = None
keras_model = None
keras_load_error = None
tflite_load_error = None


def load_tflite():
    global interpreter, input_details, output_details, tflite_load_error
    if not TFLITE_PATH.exists():
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

        interpreter = Interpreter(model_path=str(TFLITE_PATH))
        interpreter.allocate_tensors()

        # Dynamic TFLite models can expose -1 dimensions in shape_signature.
        # Resize those dimensions to the inference shape before reading tensors.
        details = interpreter.get_input_details()
        if not details:
            raise RuntimeError("TFLite model has no input tensor.")

        first = details[0]
        signature = first.get("shape_signature", first["shape"])
        signature = [int(dim) for dim in signature]
        current_shape = [int(dim) for dim in first["shape"]]

        if len(signature) == 4:
            # Support the common NHWC layout and also NCHW if channel dimension is 3.
            if signature[1] == 3 and signature[-1] != 3:
                target_shape = [1, 3, IMAGE_SIZE, IMAGE_SIZE]
            else:
                target_shape = [1, IMAGE_SIZE, IMAGE_SIZE, 3]
        elif len(signature) == 3:
            target_shape = [IMAGE_SIZE, IMAGE_SIZE, 3]
        else:
            target_shape = current_shape

        needs_resize = (
            any(dim <= 0 for dim in signature)
            or any(dim <= 0 for dim in current_shape)
            or current_shape != target_shape
        )
        if needs_resize:
            try:
                interpreter.resize_tensor_input(first["index"], target_shape, strict=False)
                interpreter.allocate_tensors()
                log.info("Resized dynamic TFLite input from %s to %s", current_shape, target_shape)
            except Exception as resize_exc:
                log.warning("Could not resize TFLite input to %s: %s", target_shape, resize_exc)
                # Re-raise only when the model still has an unresolved dynamic shape.
                refreshed = interpreter.get_input_details()[0]
                if any(int(dim) <= 0 for dim in refreshed["shape"]):
                    raise

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        log.info("TFLite model loaded: %s", TFLITE_PATH.name)
        log.info("TFLite input details: %s", input_details)
        log.info("TFLite output details: %s", output_details)
    except Exception as exc:
        tflite_load_error = str(exc)
        log.exception("Could not load TFLite model")


def load_keras_for_gradcam():
    global keras_model, keras_load_error
    if not KERAS_PATH.exists():
        keras_load_error = (
            "Original Keras model is not present. True Grad-CAM needs the original "
            "Keras model (or a TFLite model exported with intermediate activations)."
        )
        log.warning(keras_load_error)
        return
    try:
        import tensorflow as tf
        keras_model = tf.keras.models.load_model(str(KERAS_PATH), compile=False)
        log.info("Keras model loaded for Grad-CAM: %s", KERAS_PATH.name)
    except Exception as exc:
        keras_load_error = str(exc)
        log.exception("Could not load Keras model for Grad-CAM")


load_tflite()
load_keras_for_gradcam()


# -------- Image preprocessing and prediction --------
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Return a batch of RGB pixels normalized to [0, 1], shape (1, 224, 224, 3)."""
    image = image.convert("L").convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def prepare_for_tflite(batch: np.ndarray) -> np.ndarray:
    if interpreter is None or not input_details:
        raise RuntimeError(tflite_load_error or "TFLite model is not loaded.")

    detail = input_details[0]
    expected_shape = tuple(int(x) for x in detail["shape"])
    expected_dtype = detail["dtype"]

    # The interpreter input is resized at startup for dynamic models. Use the
    # resolved tensor dimensions here rather than the original shape_signature.
    if len(expected_shape) == 4:
        if expected_shape[1] == 3 and expected_shape[-1] != 3:  # NCHW
            h, w = expected_shape[2], expected_shape[3]
            channels_first = True
        else:  # NHWC
            h, w = expected_shape[1], expected_shape[2]
            channels_first = False

        if h > 0 and w > 0 and batch.shape[1:3] != (h, w):
            resized = Image.fromarray(np.uint8(np.clip(batch[0] * 255, 0, 255)))
            resized = resized.resize((w, h), Image.Resampling.BILINEAR)
            batch = np.expand_dims(np.asarray(resized, dtype=np.float32) / 255.0, axis=0)

        if channels_first:
            batch = np.transpose(batch, (0, 3, 1, 2))
    elif len(expected_shape) == 3:
        h, w = expected_shape[0], expected_shape[1]
        if h > 0 and w > 0 and batch.shape[1:3] != (h, w):
            resized = Image.fromarray(np.uint8(np.clip(batch[0] * 255, 0, 255)))
            resized = resized.resize((w, h), Image.Resampling.BILINEAR)
            batch = np.asarray(resized, dtype=np.float32) / 255.0

    if np.issubdtype(expected_dtype, np.integer):
        scale, zero_point = detail.get("quantization", (0.0, 0))
        if scale and scale > 0:
            arr = np.round(batch / scale + zero_point)
        else:
            arr = np.round(batch * 255.0)
        info = np.iinfo(expected_dtype)
        return np.clip(arr, info.min, info.max).astype(expected_dtype)

    return batch.astype(expected_dtype)


def normalize_probabilities(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Model returned non-finite prediction values.")

    # If output resembles logits, convert logits to probabilities.
    if np.any(values < 0) or not np.isclose(np.sum(values), 1.0, atol=0.03):
        values = values - np.max(values)
        exp_values = np.exp(values)
        values = exp_values / np.sum(exp_values)
    else:
        values = values / max(np.sum(values), 1e-12)

    if len(values) != len(CLASS_NAMES):
        raise RuntimeError(
            f"Model returned {len(values)} outputs, but CLASS_NAMES has "
            f"{len(CLASS_NAMES)} labels. Update CLASS_NAMES to match your trained model."
        )
    return values


def predict(batch: np.ndarray) -> np.ndarray:
    if interpreter is None:
        raise RuntimeError(tflite_load_error or "TFLite model is not loaded.")
    model_input = prepare_for_tflite(batch)
    interpreter.set_tensor(input_details[0]["index"], model_input)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_details[0]["index"])
    return normalize_probabilities(raw)


def make_decision(prediction: str, confidence: float) -> Dict[str, Any]:
    # This is an application-level confidence rule, not a trained quality/safety rule.
    if confidence >= 0.80:
        status = "High confidence"
        recommendation = f"The model strongly predicts the {prediction} pattern."
    elif confidence >= 0.55:
        status = "Moderate confidence"
        recommendation = f"The likely pattern is {prediction}; review the heatmap and wafer image."
    else:
        status = "Low confidence — manual review recommended"
        recommendation = "The model is uncertain. Do not rely on this prediction alone."
    return {
        "status": status,
        "recommendation": recommendation,
        "confidence_threshold_note": "Thresholds are heuristic and should be validated on your dataset."
    }


# -------- True Grad-CAM (requires original Keras model) --------
def generate_gradcam(batch: np.ndarray, output_filename: str) -> Optional[str]:
    if keras_model is None:
        log.warning("Grad-CAM skipped: %s", keras_load_error)
        return None

    try:
        import tensorflow as tf

        conv_layers = [
            layer for layer in keras_model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        if not conv_layers:
            log.warning("Grad-CAM skipped: no Conv2D layer found in Keras model.")
            return None

        last_conv = conv_layers[-1]
        grad_model = tf.keras.Model(
            inputs=keras_model.inputs,
            outputs=[last_conv.output, keras_model.output]
        )
        tensor = tf.convert_to_tensor(batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(tensor, training=False)
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            elif isinstance(predictions, dict):
                predictions = next(iter(predictions.values()))
            class_index = tf.argmax(predictions[0])
            class_score = predictions[:, class_index]

        gradients = tape.gradient(class_score, conv_outputs)
        if gradients is None:
            log.warning("Grad-CAM skipped: gradients unavailable.")
            return None

        weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)
        heatmap = heatmap / (max_value + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()

        heat_img = Image.fromarray(np.uint8(np.clip(heatmap * 255, 0, 255)))
        heat_img = heat_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        heat = np.asarray(heat_img, dtype=np.float32) / 255.0

        # Overlay heatmap on the original grayscale image. Red/yellow indicates
        # relative activation; this is an explanation, not proof of causality.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        base = np.uint8(np.clip(batch[0] * 255, 0, 255))
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(base)
        plt.imshow(heat, cmap="jet", alpha=0.40, vmin=0, vmax=1)
        plt.axis("off")
        fig.tight_layout(pad=0)
        target = RESULT_DIR / output_filename
        fig.savefig(target, bbox_inches="tight", pad_inches=0, dpi=140)
        plt.close(fig)
        return f"/gradcam/{output_filename}"
    except Exception:
        log.exception("Grad-CAM generation failed")
        return None


# -------- NVIDIA chat assistant (OpenAI-compatible endpoint) --------
def ask_nvidia(messages):
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not configured in Render environment variables."
        )

    from openai import OpenAI
    client = OpenAI(
        base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=api_key,
        timeout=45.0,
    )
    model_name = os.getenv(
        "NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=700,
    )
    return response.choices[0].message.content or "I could not generate a response."


def make_initial_analysis(prediction, confidence, probabilities, decision):
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return (
            f"Predicted pattern: {prediction}. Confidence: {confidence:.1%}. "
            f"Decision: {decision['status']}. Add NVIDIA_API_KEY in Render to enable "
            "AI-generated analysis and chat."
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You are WaferGPT, an assistant for semiconductor wafer-map defect "
                "classification. Use only the provided model outputs. Do not claim the "
                "prediction is certain or infer manufacturing root cause without evidence. "
                "Explain briefly in plain language."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Model prediction: {prediction}\n"
                f"Confidence: {confidence:.1%}\n"
                f"Class probabilities: {probabilities}\n"
                f"Decision: {decision}\n"
                "Give a concise interpretation, limitations, and suggested next checks."
            ),
        },
    ]
    try:
        return ask_nvidia(messages)
    except Exception as exc:
        log.exception("Initial NVIDIA analysis failed")
        return (
            f"Prediction: {prediction} ({confidence:.1%}). Decision: {decision['status']}. "
            f"AI analysis could not be generated: {str(exc)[:250]}"
        )


# -------- API routes --------
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
        "keras_model_loaded_for_gradcam": keras_model is not None,
        "gradcam_note": keras_load_error if keras_model is None else "Ready",
        "tflite_error": tflite_load_error,
    }


@app.get("/gradcam/{filename}")
def get_gradcam(filename: str):
    # Prevent path traversal; only serve files created in RESULT_DIR.
    safe_name = Path(filename).name
    target = RESULT_DIR / safe_name
    if not target.exists():
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
        raise HTTPException(status_code=400, detail="Could not read this image. Upload a valid PNG/JPG.")

    request_id = uuid.uuid4().hex[:12]
    saved_name = f"{request_id}.png"
    image.save(UPLOAD_DIR / saved_name, format="PNG")

    try:
        log.info("[%s] Starting TFLite prediction", request_id)
        probs = predict(batch)
        idx = int(np.argmax(probs))
        prediction = CLASS_NAMES[idx]
        confidence = float(probs[idx])
        probabilities = {
            CLASS_NAMES[i]: round(float(probs[i]), 6)
            for i in range(len(CLASS_NAMES))
        }
        decision = make_decision(prediction, confidence)

        log.info("[%s] Prediction=%s confidence=%.4f", request_id, prediction, confidence)
        gradcam_name = f"{request_id}_gradcam.png"
        gradcam_url = generate_gradcam(batch, gradcam_name)

        log.info("[%s] Starting assistant analysis", request_id)
        llm_analysis = make_initial_analysis(
            prediction, confidence, probabilities, decision
        )

        return {
            "success": True,
            "request_id": request_id,
            "filename": saved_name,
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
                else "True Grad-CAM is unavailable. Add a compatible backend/wafer_model.keras file."
            ),
            "llm_analysis": llm_analysis,
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("[%s] Prediction pipeline failed", request_id)
        raise HTTPException(status_code=500, detail=f"Prediction pipeline failed: {str(exc)[:300]}")


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
    messages = [
        {
            "role": "system",
            "content": (
                "You are WaferGPT, a helpful semiconductor wafer-map assistant. Answer "
                "questions using the provided prediction context. Explain class probabilities "
                "and Grad-CAM carefully. Never present model output as a guaranteed diagnosis "
                "or claim the heatmap proves causality. If there is no prediction context, ask "
                "the user to upload an image first.\n\nContext:\n" + context
            ),
        }
    ]

    # Keep only recent valid user/assistant turns to control request size.
    for item in (body.history or [])[-8:]:
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"}:
            content = item.get("content")
            if isinstance(content, str) and content.strip():
                messages.append({"role": item["role"], "content": content[:2000]})
    messages.append({"role": "user", "content": message})

    try:
        answer = ask_nvidia(messages)
        return {"success": True, "reply": answer}
    except Exception as exc:
        log.exception("Chat assistant failed")
        raise HTTPException(
            status_code=502,
            detail=f"Chat assistant unavailable: {str(exc)[:250]}"
        )
