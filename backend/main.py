import asyncio
import uuid
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from llm import analyze_wafer, client as gemini_client


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="WaferGPT",
    description="AI Semiconductor Wafer Defect Intelligence",
    version="3.1.0"
)


# ============================================================
# CORS
# ============================================================

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
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


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
    "none"
]


# ============================================================
# GEMINI MODEL
# ============================================================

GEMINI_MODEL = "gemini-3.7-flash"


# ============================================================
# LOAD MODEL
# ============================================================

print("")
print("==========================================")
print("        WAFER GPT MODEL LOADING")
print("==========================================")
print("")

print("Loading model from:")
print(MODEL_PATH)

try:

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully")

    print(
        "Input shape :",
        model.input_shape
    )

    print(
        "Output shape:",
        model.output_shape
    )

except Exception as e:

    print(
        "MODEL LOADING ERROR:",
        e
    )

    raise e


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_wafer(image: Image.Image):
    """
    Preprocess uploaded wafer-map images.

    Training representation:
        raw values 0 / 1 / 2
        ->
        0 / 127 / 254
        ->
        224x224
        ->
        grayscale to RGB
        ->
        /255

    Uploaded PNG/JPG files can contain anti-aliased
    grayscale values, so we map them to the nearest
    WM-811K-style intensity level.
    """

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    image = image.convert("L")

    image = np.array(
        image,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Normalize image range if required
    # --------------------------------------------------------

    min_value = int(image.min())
    max_value = int(image.max())

    if max_value > min_value:

        # If image already looks like a WM-811K map
        # containing approximately 0/127/254, preserve it.

        unique_values = np.unique(image)

        has_wafer_levels = (
            np.any(np.abs(unique_values - 0) <= 5)
            and
            np.any(np.abs(unique_values - 127) <= 10)
            and
            np.any(np.abs(unique_values - 254) <= 5)
        )

        if not has_wafer_levels:

            image_float = image.astype(
                np.float32
            )

            image_float = (
                (image_float - min_value)
                /
                (max_value - min_value)
            ) * 254.0

            image = image_float.astype(
                np.uint8
            )

    # --------------------------------------------------------
    # Quantize to WM-811K-style levels
    #
    # 0   -> background
    # 127 -> normal wafer/die level
    # 254 -> high intensity level
    # --------------------------------------------------------

    distances = np.stack(
        [
            np.abs(image.astype(np.int16) - 0),
            np.abs(image.astype(np.int16) - 127),
            np.abs(image.astype(np.int16) - 254)
        ],
        axis=-1
    )

    nearest_level = np.argmin(
        distances,
        axis=-1
    )

    wafer_map = np.zeros_like(
        image,
        dtype=np.uint8
    )

    wafer_map[nearest_level == 0] = 0
    wafer_map[nearest_level == 1] = 127
    wafer_map[nearest_level == 2] = 254

    # --------------------------------------------------------
    # Resize EXACTLY like training
    # --------------------------------------------------------

    wafer_map = cv2.resize(
        wafer_map,
        (224, 224),
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # Grayscale -> RGB
    # --------------------------------------------------------

    wafer_map = cv2.cvtColor(
        wafer_map,
        cv2.COLOR_GRAY2RGB
    )

    # --------------------------------------------------------
    # Float normalization
    # --------------------------------------------------------

    wafer_map = wafer_map.astype(
        np.float32
    ) / 255.0

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    wafer_map = np.expand_dims(
        wafer_map,
        axis=0
    )

    return wafer_map


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    image,
    predicted_index,
    output_path
):

    try:

        # ----------------------------------------------------
        # FIND LAST CONVOLUTIONAL LAYER
        # ----------------------------------------------------

        target_layer = None

        for layer in reversed(
            model.layers
        ):

            if isinstance(
                layer,
                (
                    tf.keras.layers.Conv2D,
                    tf.keras.layers.DepthwiseConv2D
                )
            ):

                target_layer = layer
                break

        if target_layer is None:

            print(
                "No convolutional layer found."
            )

            return False

        print(
            "Grad-CAM target layer:",
            target_layer.name
        )

        # ----------------------------------------------------
        # GRADIENT MODEL
        # ----------------------------------------------------

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                target_layer.output,
                model.output
            ]
        )

        # ----------------------------------------------------
        # GRADIENT CALCULATION
        # ----------------------------------------------------

        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(
                image,
                training=False
            )

            if isinstance(
                conv_outputs,
                (list, tuple)
            ):

                conv_outputs = conv_outputs[0]

            if isinstance(
                predictions,
                (list, tuple)
            ):

                predictions = predictions[0]

            class_output = predictions[
                :,
                predicted_index
            ]

        grads = tape.gradient(
            class_output,
            conv_outputs
        )

        if grads is None:

            print(
                "Could not calculate gradients."
            )

            return False

        # ----------------------------------------------------
        # GLOBAL AVERAGE POOLING OF GRADIENTS
        # ----------------------------------------------------

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        # ----------------------------------------------------
        # CREATE HEATMAP
        # ----------------------------------------------------

        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        max_value = tf.reduce_max(
            heatmap
        )

        heatmap = tf.where(
            max_value > 0,
            heatmap / max_value,
            heatmap
        )

        heatmap = heatmap.numpy()

        # ----------------------------------------------------
        # ORIGINAL IMAGE
        # ----------------------------------------------------

        original = image[0]

        if hasattr(
            original,
            "numpy"
        ):

            original = original.numpy()

        original = (
            original * 255
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )

        original = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2BGR
        )

        # ----------------------------------------------------
        # RESIZE HEATMAP
        # ----------------------------------------------------

        heatmap = cv2.resize(
            heatmap,
            (
                original.shape[1],
                original.shape[0]
            )
        )

        heatmap_uint8 = (
            heatmap * 255
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )

        colored_heatmap = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        # ----------------------------------------------------
        # OVERLAY
        # ----------------------------------------------------

        overlay = cv2.addWeighted(
            original,
            0.6,
            colored_heatmap,
            0.4,
            0
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        success = cv2.imwrite(
            str(output_path),
            overlay
        )

        if not success:

            print(
                "Failed to save Grad-CAM."
            )

            return False

        print(
            "Grad-CAM generated:",
            output_path
        )

        return True

    except Exception as e:

        print(
            "Grad-CAM ERROR:",
            e
        )

        return False


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "name": "WaferGPT",
        "status": "online",
        "version": "3.1.0"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gemini": gemini_client is not None
    }


# ============================================================
# UPLOADED IMAGE
# ============================================================

@app.get("/uploads/{filename}")
async def get_uploaded_image(
    filename: str
):

    file_path = (
        UPLOAD_DIR /
        filename
    )

    if not file_path.exists():

        return {
            "error": "Image not found"
        }

    return FileResponse(
        file_path
    )


# ============================================================
# GRAD-CAM FILE
# ============================================================

@app.get("/gradcam/{filename}")
async def get_gradcam(
    filename: str
):

    file_path = (
        RESULT_DIR /
        filename
    )

    if not file_path.exists():

        return {
            "error": "Grad-CAM not found"
        }

    return FileResponse(
        file_path
    )


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    print("")
    print("==========================================")
    print("NEW WAFER ANALYSIS")
    print("==========================================")

    # --------------------------------------------------------
    # VALIDATE FILE
    # --------------------------------------------------------

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp"
    }

    original_name = (
        file.filename
        or
        "wafer.png"
    )

    extension = Path(
        original_name
    ).suffix.lower()

    if extension not in allowed_extensions:

        return {
            "success": False,
            "error": "Unsupported image format."
        }

    # --------------------------------------------------------
    # UNIQUE FILE NAME
    # --------------------------------------------------------

    unique_id = uuid.uuid4().hex[:12]

    saved_filename = (
        f"{unique_id}{extension}"
    )

    file_path = (
        UPLOAD_DIR /
        saved_filename
    )

    # --------------------------------------------------------
    # SAVE IMAGE
    # --------------------------------------------------------

    contents = await file.read()

    with open(
        file_path,
        "wb"
    ) as buffer:

        buffer.write(
            contents
        )

    print(
        "Image saved:",
        file_path
    )

    # --------------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            file_path
        )

        print(
            "Original image size:",
            image.size
        )

        print(
            "Original image mode:",
            image.mode
        )

        image = image.convert(
            "RGB"
        )

    except Exception as e:

        print(
            "IMAGE ERROR:",
            e
        )

        return {
            "success": False,
            "error": "Invalid image file."
        }

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    processed_image = preprocess_wafer(
        image
    )

    print(
        "Processed image shape:",
        processed_image.shape
    )

    print(
        "Processed image min:",
        float(processed_image.min())
    )

    print(
        "Processed image max:",
        float(processed_image.max())
    )

    # --------------------------------------------------------
    # CNN PREDICTION
    # --------------------------------------------------------

    try:

        predictions = model.predict(
            processed_image,
            verbose=0
        )

    except Exception as e:

        print(
            "MODEL PREDICTION ERROR:",
            e
        )

        return {
            "success": False,
            "error": "Model prediction failed."
        }

    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # VALIDATE OUTPUT
    # --------------------------------------------------------

    if predictions.ndim != 2:

        print(
            "Unexpected model output:",
            predictions.shape
        )

        return {
            "success": False,
            "error": "Unexpected model output shape."
        }

    predictions = predictions[0]

    if len(predictions) != len(CLASS_NAMES):

        print(
            "Class count mismatch:",
            len(predictions),
            len(CLASS_NAMES)
        )

        return {
            "success": False,
            "error": "Model output classes do not match CLASS_NAMES."
        }

    # --------------------------------------------------------
    # NUMERIC SAFETY
    # --------------------------------------------------------

    predictions = np.nan_to_num(
        predictions,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # PREDICTED CLASS
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(
            predictions
        )
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[
            predicted_index
        ]
    )

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    probabilities = {
        CLASS_NAMES[i]:
        float(predictions[i])
        for i in range(
            len(CLASS_NAMES)
        )
    }

    # --------------------------------------------------------
    # LOG RESULTS
    # --------------------------------------------------------

    print(
        "Prediction:",
        predicted_class
    )

    print(
        "Confidence:",
        confidence * 100
    )

    print(
        "Probabilities:"
    )

    for class_name, probability in probabilities.items():

        print(
            f"  {class_name}: "
            f"{probability * 100:.2f}%"
        )

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    gradcam_filename = (
        f"gradcam_{unique_id}.jpg"
    )

    gradcam_path = (
        RESULT_DIR /
        gradcam_filename
    )

    gradcam_success = generate_gradcam(
        processed_image,
        predicted_index,
        gradcam_path
    )

    gradcam_url = None

    if gradcam_success:

        gradcam_url = (
            f"/gradcam/{gradcam_filename}"
        )

    # --------------------------------------------------------
    # GEMINI ANALYSIS
    # --------------------------------------------------------

    llm_analysis = ""
    llm_enabled = False
    llm_error = None

    if gemini_client is not None:

        try:

            print(
                "Sending image to Gemini..."
            )

            llm_analysis = await asyncio.to_thread(
                analyze_wafer,
                predicted_class,
                confidence * 100,
                probabilities,
                str(file_path)
            )

            if llm_analysis:

                llm_enabled = True

                print(
                    "Gemini analysis generated."
                )

        except Exception as e:

            llm_error = str(e)

            print(
                "Gemini analysis error:",
                e
            )

    else:

        llm_error = (
            "Gemini client is not available."
        )

        print(
            llm_error
        )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    response = {

        "success": True,

        "filename":
            saved_filename,

        "prediction":
            predicted_class,

        "confidence":
            confidence * 100,

        "probabilities":
            probabilities,

        "gradcam_url":
            gradcam_url,

        "llm_analysis":
            llm_analysis,

        "llm_enabled":
            llm_enabled

    }

    if llm_error:

        response[
            "llm_error"
        ] = llm_error

    print(
        "Analysis complete."
    )

    return response


# ============================================================
# WAFER GPT CHAT
# ============================================================

@app.post("/chat")
async def chat(
    request: dict
):

    print("")
    print("==========================================")
    print("WAFER GPT CHAT REQUEST")
    print("==========================================")

    try:

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        user_message = request.get(
            "message",
            ""
        )

        if not isinstance(
            user_message,
            str
        ):

            user_message = str(
                user_message
            )

        user_message = user_message.strip()

        if not user_message:

            return {
                "success": False,
                "error": "Message is required."
            }

        # ----------------------------------------------------
        # CURRENT WAFER CONTEXT
        # ----------------------------------------------------

        prediction = request.get(
            "prediction",
            "Unknown"
        )

        confidence = request.get(
            "confidence",
            0
        )

        probabilities = request.get(
            "probabilities",
            {}
        )

        llm_analysis = request.get(
            "llm_analysis",
            ""
        )

        # ----------------------------------------------------
        # SAFE CONFIDENCE
        # ----------------------------------------------------

        try:

            confidence = float(
                confidence
            )

        except Exception:

            confidence = 0.0

        # ----------------------------------------------------
        # SAFE PROBABILITIES
        # ----------------------------------------------------

        if not isinstance(
            probabilities,
            dict
        ):

            probabilities = {}

        # ----------------------------------------------------
        # CHECK GEMINI
        # ----------------------------------------------------

        if gemini_client is None:

            return {
                "success": False,
                "error": "Gemini client is not available."
            }

        # ----------------------------------------------------
        # CHAT PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are WAFER GPT, an AI semiconductor wafer
defect analysis assistant.

Your job is to help engineers understand wafer
defect classification results.

CURRENT WAFER
=============

Prediction:
{prediction}

Confidence:
{confidence:.2f}%

Class Probabilities:
{probabilities}

Previous AI Analysis:
{llm_analysis}

USER QUESTION
=============

{user_message}

RESPONSE RULES
==============

1. Answer the user's exact question.

2. When the question is about the current wafer,
   use the wafer context provided above.

3. Explain semiconductor wafer defect concepts clearly
   and professionally.

4. Never claim that an AI prediction is a confirmed
   engineering diagnosis.

5. Possible root causes must be described as hypotheses
   requiring engineering verification.

6. Do not invent process measurements, equipment data,
   wafer history, manufacturing conditions, or sensor data.

7. If confidence is discussed, explain that it represents
   the model's predicted probability and should not
   automatically be treated as engineering certainty.

8. If confidence is low, explicitly acknowledge model
   uncertainty.

9. Use concise paragraphs and bullet points where useful.

10. If the user asks about why a classification was made,
    explain the likely visual pattern associated with
    that classification based only on the provided context.

11. If the user asks for inspection recommendations,
    provide practical engineering inspection steps.

12. If the user asks an unrelated general question,
    answer normally.

13. Act like a semiconductor AI assistant rather than
    a generic chatbot.

USER:
{user_message}
"""

        print(
            "Sending chat request to Gemini..."
        )

        # ----------------------------------------------------
        # GEMINI REQUEST
        # ----------------------------------------------------

        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=GEMINI_MODEL,
            contents=prompt
        )

        # ----------------------------------------------------
        # EXTRACT RESPONSE
        # ----------------------------------------------------

        answer = getattr(
            response,
            "text",
            None
        )

        if not answer:

            answer = (
                "I could not generate a response "
                "for this question."
            )

        print(
            "Chat response generated successfully."
        )

        return {

            "success": True,

            "response":
                answer

        }

    except Exception as e:

        print(
            "CHAT ERROR:",
            e
        )

        return {

            "success": False,

            "error":
                str(e)

        }
