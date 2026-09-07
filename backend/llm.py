import os
import time
from pathlib import Path

from google import genai
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".venv" / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")


# ============================================================
# GEMINI CONFIG
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None


# Primary model
PRIMARY_MODEL = "gemini-3.6-flash"

# Backup model
FALLBACK_MODEL = "gemini-3.7-flash"


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_wafer(
    predicted_class,
    confidence,
    probabilities,
    image_path
):

    if client is None:
        return "Gemini analysis unavailable: GEMINI_API_KEY is not configured."

    prompt = f"""
You are WAFER GPT, an AI semiconductor wafer defect
analysis assistant.

A CNN model analyzed the wafer image.

CNN PREDICTION:
{predicted_class}

CNN CONFIDENCE:
{confidence:.2f}%

CLASS PROBABILITIES:
{probabilities}

Analyze the uploaded wafer image visually and combine
the visual evidence with the CNN prediction.

IMPORTANT:
- Do not claim a confirmed manufacturing diagnosis.
- Treat root causes as possible hypotheses.
- Clearly distinguish AI prediction from engineering verification.
- Give practical semiconductor inspection recommendations.

Return the analysis in exactly this structure:

DEFECT ANALYSIS
Explain the predicted defect pattern.

VISUAL EVIDENCE
Describe what is visible in the wafer image.

WHY THIS PREDICTION
Explain why the CNN prediction is reasonable.

POSSIBLE ROOT CAUSES
List likely manufacturing/process causes.

RECOMMENDED INSPECTION
Give practical engineering checks.

SEVERITY
Give LOW, MEDIUM, or HIGH and explain briefly.

ENGINEER SUMMARY
Give a concise professional summary.
"""


    # ========================================================
    # UPLOAD IMAGE
    # ========================================================

    try:

        uploaded_file = client.files.upload(
            file=image_path
        )

    except Exception as e:

        print("Gemini image upload error:", e)

        return (
            "Gemini image upload failed. "
            "Please try the analysis again."
        )


    # ========================================================
    # RETRY FUNCTION
    # ========================================================

    def call_model(model_name):

        max_retries = 3

        for attempt in range(max_retries):

            try:

                print(
                    f"Gemini request → {model_name} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )


                response = client.models.generate_content(

                    model=model_name,

                    contents=[
                        prompt,
                        uploaded_file
                    ]
                )


                if response and response.text:

                    print(
                        f"Gemini success → {model_name}"
                    )

                    return response.text


                raise RuntimeError(
                    "Gemini returned an empty response."
                )


            except Exception as e:

                error_text = str(e)

                print(
                    f"Gemini error [{model_name}] "
                    f"attempt {attempt + 1}: {error_text}"
                )


                # Retry temporary server errors
                if (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "500" in error_text
                    or "502" in error_text
                    or "504" in error_text
                ):

                    if attempt < max_retries - 1:

                        wait_time = 2 ** attempt

                        print(
                            f"Temporary Gemini issue. "
                            f"Retrying in {wait_time} seconds..."
                        )

                        time.sleep(wait_time)

                        continue


                # Non-temporary error
                raise


        return None


    # ========================================================
    # PRIMARY MODEL
    # ========================================================

    try:

        result = call_model(
            PRIMARY_MODEL
        )

        if result:
            return result


    except Exception as primary_error:

        print(
            "Primary Gemini model failed:",
            primary_error
        )


    # ========================================================
    # FALLBACK MODEL
    # ========================================================

    print(
        f"Switching to fallback model: "
        f"{FALLBACK_MODEL}"
    )


    try:

        result = call_model(
            FALLBACK_MODEL
        )

        if result:
            return result


    except Exception as fallback_error:

        print(
            "Fallback Gemini model failed:",
            fallback_error
        )


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return """
GEMINI ANALYSIS TEMPORARILY UNAVAILABLE

The CNN prediction and Grad-CAM analysis were completed
successfully, but the Gemini engineering analysis could
not be generated at this moment.

Please try the analysis again shortly.
"""