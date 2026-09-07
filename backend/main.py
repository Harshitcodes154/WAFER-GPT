import asyncio

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from PIL import Image

import tensorflow as tf
import numpy as np
import cv2
import uuid

from llm import analyze_wafer, client as gemini_client


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="WaferGPT",
    description="AI Semiconductor Wafer Defect Intelligence",
    version="3.0.0"
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

UPLOAD_DIR.mkdir(exist_ok=True)

RESULT_DIR.mkdir(exist_ok=True)


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
# LOAD MODEL
# ============================================================

print("")
print("==========================================")
print("        WAFER GPT MODEL LOADING")
print("==========================================")
print("")

print(
    "Loading model from:",
    MODEL_PATH
)

try:

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully"
    )

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

def preprocess_wafer(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image = np.array(
        image,
        dtype=np.float32
    )

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    image,
    predicted_index,
    output_path
):

    try:

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


        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                target_layer.output,
                model.output
            ]
        )


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


        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )


        conv_outputs = conv_outputs[0]


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
        ).astype(
            np.uint8
        )


        original = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2BGR
        )


        # ----------------------------------------------------
        # HEATMAP
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

        cv2.imwrite(
            str(output_path),
            overlay
        )


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
        "version": "3.0.0"
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


    # ========================================================
    # VALIDATE FILE
    # ========================================================

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp"
    }


    original_name = (
        file.filename or
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


    # ========================================================
    # UNIQUE FILE NAME
    # ========================================================

    unique_id = uuid.uuid4().hex[:12]


    saved_filename = (
        f"{unique_id}{extension}"
    )


    file_path = (
        UPLOAD_DIR /
        saved_filename
    )


    # ========================================================
    # SAVE IMAGE
    # ========================================================

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


    # ========================================================
    # OPEN IMAGE
    # ========================================================

    try:

        image = Image.open(
            file_path
        ).convert("RGB")

    except Exception as e:

        print(
            "IMAGE ERROR:",
            e
        )


        return {
            "success": False,
            "error": "Invalid image file."
        }


    # ========================================================
    # PREPROCESS
    # ========================================================

    processed_image = preprocess_wafer(
        image
    )


    # ========================================================
    # CNN PREDICTION
    # ========================================================

    predictions = model.predict(
        processed_image,
        verbose=0
    )


    predictions = np.asarray(
        predictions
    )[0]


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


    probabilities = {

        CLASS_NAMES[i]:
            float(predictions[i])

        for i in range(
            len(CLASS_NAMES)
        )

    }


    print(
        "Prediction:",
        predicted_class
    )


    print(
        "Confidence:",
        confidence * 100
    )


    # ========================================================
    # GRAD-CAM
    # ========================================================

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


    # ========================================================
    # GEMINI ANALYSIS
    # ========================================================

    llm_analysis = ""

    llm_enabled = False

    llm_error = None


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


    except Exception as e:

        llm_error = str(e)


        print(
            "Gemini analysis error:",
            e
        )


    # ========================================================
    # RESPONSE
    # ========================================================

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
        # GET USER MESSAGE
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

3. Explain semiconductor defect concepts clearly
   and professionally.

4. Do not claim that an AI prediction is a confirmed
   engineering diagnosis.

5. Possible root causes must be described as hypotheses
   that require engineering verification.

6. Do not invent process measurements, equipment data,
   wafer history, or manufacturing conditions.

7. If confidence is discussed, explain that confidence
   represents the model's predicted probability and
   should not automatically be treated as certainty.

8. Use concise paragraphs and bullet points where useful.

9. If the user asks an unrelated general question,
   answer normally.

10. Act like a semiconductor AI assistant rather than
    a generic chatbot.

USER:
{user_message}
"""


        print(
            "Sending chat request to Gemini..."
        )


        # ----------------------------------------------------
        # GEMINI CHAT REQUEST
        # ----------------------------------------------------

        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model="gemini-3.7-flash",
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