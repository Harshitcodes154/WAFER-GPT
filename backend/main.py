
import asyncio
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from llm import analyze_wafer, client as gemini_client


# ============================================================
# APP
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
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "wafer_model.keras"
TFLITE_MODEL_PATH = BASE_DIR / "wafer_model_dynamic.tflite"

UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

GEMINI_MODEL = "gemini-3.7-flash"


# ============================================================
# CLASS NAMES — KEEP TRAINING ORDER
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
# LOAD ORIGINAL KERAS MODEL FOR GRAD-CAM
# ============================================================

print("Loading original Keras model for Grad-CAM...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Original Keras model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False,
)

print("Keras model loaded:", model.input_shape, model.output_shape)


# ============================================================
# LOAD OPTIMIZED TFLITE MODEL FOR PREDICTION
# ============================================================

print("Loading optimized TFLite model...")

if not TFLITE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"TFLite model not found: {TFLITE_MODEL_PATH}"
    )

interpreter = tf.lite.Interpreter(
    model_path=str(TFLITE_MODEL_PATH)
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

if tuple(input_details[0]["shape"]) != (1, 224, 224, 3):
    raise ValueError(
        f"Unexpected TFLite input shape: "
        f"{input_details[0]['shape']}"
    )

if int(output_details[0]["shape"][-1]) != len(CLASS_NAMES):
    raise ValueError(
        f"TFLite output must contain {len(CLASS_NAMES)} classes."
    )

# One interpreter is shared by all requests.
# Lock it while setting inputs and invoking inference.
tflite_lock = threading.Lock()

print("TFLite model loaded successfully.")
print("TFLite input:", input_details[0]["shape"])
print("TFLite output:", output_details[0]["shape"])


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_wafer(image: Image.Image) -> np.ndarray:
    """
    Preserve the existing WM-811K-style preprocessing:
    grayscale -> intensity levels 0/127/254 -> resize
    -> RGB -> float32 / 255 -> batch dimension.
    """

    image = image.convert("L")
    image = np.asarray(image, dtype=np.uint8)

    min_value = int(image.min())
    max_value = int(image.max())

    if max_value > min_value:
        unique_values = np.unique(image)

        has_wafer_levels = (
            np.any(np.abs(unique_values.astype(np.int16) - 0) <= 5)
            and np.any(
                np.abs(unique_values.astype(np.int16) - 127) <= 10
            )
            and np.any(
                np.abs(unique_values.astype(np.int16) - 254) <= 5
            )
        )

        if not has_wafer_levels:
            image_float = image.astype(np.float32)
            image_float = (
                (image_float - min_value)
                / (max_value - min_value)
            ) * 254.0

            image = image_float.astype(np.uint8)

    image_int = image.astype(np.int16)

    distances = np.stack(
        [
            np.abs(image_int - 0),
            np.abs(image_int - 127),
            np.abs(image_int - 254),
        ],
        axis=-1,
    )

    nearest_level = np.argmin(distances, axis=-1)

    wafer_map = np.zeros_like(image, dtype=np.uint8)
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

def predict_tflite(processed_image: np.ndarray) -> np.ndarray:
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

    if result.ndim != 2 or result.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Unexpected TFLite output shape: {result.shape}"
        )

    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return result[0]


# ============================================================
# GRAD-CAM — ORIGINAL KERAS MODEL
# ============================================================

def generate_gradcam(
    image: np.ndarray,
    predicted_index: int,
    output_path: Path,
) -> bool:
    try:
        target_layer = None

        for layer in reversed(model.layers):
            if isinstance(
                layer,
                (
                    tf.keras.layers.Conv2D,
                    tf.keras.layers.DepthwiseConv2D,
                ),
            ):
                target_layer = layer
                break

        if target_layer is None:
            print("Grad-CAM: no convolutional layer found.")
            return False

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(
                image,
                training=False,
            )

            class_output = predictions[:, predicted_index]

        grads = tape.gradient(class_output, conv_outputs)

        if grads is None:
            print("Grad-CAM: gradients unavailable.")
            return False

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2),
        )

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads,
            axis=-1,
        )

        heatmap = tf.maximum(heatmap, 0)

        max_value = tf.reduce_max(heatmap)
        heatmap = tf.where(
            max_value > 0,
            heatmap / tf.maximum(max_value, 1e-8),
            heatmap,
        ).numpy()

        original = (image[0] * 255.0).clip(
            0, 255
        ).astype(np.uint8)

        original = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2BGR,
        )

        heatmap = cv2.resize(
            heatmap,
            (original.shape[1], original.shape[0]),
        )

        heatmap_uint8 = (heatmap * 255).clip(
            0, 255
        ).astype(np.uint8)

        colored_heatmap = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET,
        )

        overlay = cv2.addWeighted(
            original,
            0.6,
            colored_heatmap,
            0.4,
            0,
        )

        return bool(cv2.imwrite(str(output_path), overlay))

    except Exception as exc:
        print("Grad-CAM error:", exc)
        return False


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():
    return {
        "name": "WaferGPT",
        "status": "online",
        "version": "3.2.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tflite_model_loaded": interpreter is not None,
        "gemini": gemini_client is not None,
    }


# ============================================================
# SERVE UPLOADED IMAGE
# ============================================================

@app.get("/uploads/{filename}")
async def get_uploaded_image(filename: str):
    safe_name = Path(filename).name
    file_path = UPLOAD_DIR / safe_name

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(file_path)


# ============================================================
# SERVE GRAD-CAM IMAGE
# ============================================================

@app.get("/gradcam/{filename}")
async def get_gradcam(filename: str):
    safe_name = Path(filename).name
    file_path = RESULT_DIR / safe_name

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Grad-CAM not found")

    return FileResponse(file_path)


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    original_name = file.filename or "wafer.png"
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a PNG, JPG, JPEG, or WEBP image.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller.",
        )

    unique_id = uuid.uuid4().hex[:12]
    saved_filename = f"{unique_id}{extension}"
    file_path = UPLOAD_DIR / saved_filename

    try:
        with Image.open(__import__("io").BytesIO(contents)) as source:
            source.load()
            image = source.convert("RGB")

        image.save(file_path)

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported image.",
        ) from exc

    processed_image = preprocess_wafer(image)

    print("Processed image shape:", processed_image.shape)

    # Run optimized TFLite inference.
    try:
        predictions = await asyncio.to_thread(
            predict_tflite,
            processed_image,
        )
    except Exception as exc:
        print("TFLite prediction error:", exc)
        raise HTTPException(
            status_code=500,
            detail="TFLite model prediction failed.",
        ) from exc

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    probabilities = {
        CLASS_NAMES[index]: float(predictions[index])
        for index in range(len(CLASS_NAMES))
    }

    print("Prediction:", predicted_class)
    print("Top output score:", confidence)

    # Grad-CAM uses the original Keras model.
    gradcam_filename = f"gradcam_{unique_id}.jpg"
    gradcam_path = RESULT_DIR / gradcam_filename

    gradcam_success = await asyncio.to_thread(
        generate_gradcam,
        processed_image,
        predicted_index,
        gradcam_path,
    )

    gradcam_url = (
        f"/gradcam/{gradcam_filename}"
        if gradcam_success
        else None
    )

    # Gemini analysis.
    llm_analysis = ""
    llm_enabled = False
    llm_error = None

    if gemini_client is not None:
        try:
            llm_analysis = await asyncio.to_thread(
                analyze_wafer,
                predicted_class,
                confidence * 100,
                probabilities,
                str(file_path),
            )

            llm_enabled = bool(llm_analysis)

        except Exception as exc:
            llm_error = str(exc)
            print("Gemini analysis error:", exc)
    else:
        llm_error = "Gemini client is not available."

    response = {
        "success": True,
        "filename": saved_filename,
        "prediction": predicted_class,
        "confidence": confidence * 100,
        "probabilities": probabilities,
        "gradcam_url": gradcam_url,
        "llm_analysis": llm_analysis,
        "llm_enabled": llm_enabled,
    }

    if llm_error:
        response["llm_error"] = llm_error

    return response


# ============================================================
# WAFER GPT CHAT
# ============================================================

@app.post("/chat")
async def chat(request: dict):
    user_message = str(request.get("message", "")).strip()

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="Message is required.",
        )

    if gemini_client is None:
        raise HTTPException(
            status_code=503,
            detail="Gemini client is not available.",
        )

    prediction = request.get("prediction", "Unknown")

    try:
        confidence = float(request.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    probabilities = request.get("probabilities", {})
    if not isinstance(probabilities, dict):
        probabilities = {}

    llm_analysis = request.get("llm_analysis", "")

    prompt = f"""
You are WAFER GPT, an AI semiconductor wafer defect
analysis assistant.

Current wafer prediction: {prediction}
Model output score: {confidence:.2f}%
Class probabilities: {probabilities}
Previous AI analysis: {llm_analysis}

User question:
{user_message}

Rules:
- Answer the user's exact question.
- Explain semiconductor wafer defects clearly.
- Never present an AI prediction as a confirmed diagnosis.
- Describe possible root causes as hypotheses requiring engineering verification.
- Do not invent manufacturing data or measurements.
- Explain uncertainty when discussing model scores.
- Use concise paragraphs and bullets where helpful.
"""

    try:
        result = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=GEMINI_MODEL,
            contents=prompt,
        )

        answer = getattr(result, "text", None)

        if not answer:
            answer = "I could not generate a response for this question."

        return {
            "success": True,
            "response": answer,
        }

    except Exception as exc:
        print("Chat error:", exc)
        raise HTTPException(
            status_code=500,
            detail="Unable to generate chat response.",
        ) from exc
