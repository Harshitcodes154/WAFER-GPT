<div align="center">

# ⚡ WAFER GPT

### AI-Powered Semiconductor Wafer Defect Intelligence

<img src="./assets/hero.svg" width="100%" alt="WAFER GPT"/>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=18&pause=1000&color=22D3EE&center=true&vCenter=true&width=900&lines=DETECT+%7C+CLASSIFY+%7C+EXPLAIN+%7C+UNDERSTAND;AI+POWERED+WAFER+DEFECT+INTELLIGENCE;COMPUTER+VISION+%2B+EXPLAINABLE+AI+%2B+GENERATIVE+AI;TURNING+WAFER+IMAGES+INTO+ACTIONABLE+INSIGHTS" alt="Typing SVG"/>

<br/><br/>

<a href="https://github.com/Harshitcodes154/WAFER-GPT">
<img src="https://img.shields.io/badge/GitHub-WAFER--GPT-181717?style=for-the-badge&logo=github" alt="GitHub"/>
</a>

<img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>

<img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>

<img src="https://img.shields.io/badge/Google%20Gemini-Generative%20AI-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemini"/>

<img src="https://img.shields.io/badge/Grad--CAM-Explainable%20AI-7C3AED?style=for-the-badge" alt="Grad-CAM"/>

<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>

<br/><br/>

### 🚀 Detect. Explain. Understand.

**WAFER GPT** is an AI-powered semiconductor wafer defect intelligence platform that combines:

**Deep Learning + Computer Vision + Explainable AI + Generative AI**

to classify wafer patterns, visualize model attention using Grad-CAM, generate AI-assisted analysis, and provide contextual interaction through an AI chatbot.

</div>

---

# 🧠 What is WAFER GPT?

**WAFER GPT** is an end-to-end AI system designed for intelligent semiconductor wafer defect analysis.

Traditional classification systems generally provide only a predicted class.

WAFER GPT goes further by combining:

- 🧠 Deep Learning Classification
- 📊 Confidence & Probability Distribution
- 🔬 Grad-CAM Explainability
- 🤖 Google Gemini AI Analysis
- 💬 Context-Aware AI Chat
- ⚡ FastAPI REST Backend
- 🌐 Interactive Web Interface
- ☁️ Cloud Deployment

The objective is simple:

> **Don't just predict the defect. Explain the prediction.**

---

# 🎯 The Problem

Semiconductor wafer inspection can generate complex visual patterns that are difficult to analyze manually at scale.

A conventional classifier might return:

```text
Prediction: Scratch
Confidence: 91.42%
```

But an engineer may still want to know:

```text
Why was this prediction made?

Which region influenced the model?

What were the alternative classes?

How confident is the model?

What does this wafer pattern represent?

What could potentially cause such a pattern?

How should the result be interpreted?
```

WAFER GPT addresses these questions by combining classification, explainability and generative AI into a single workflow.

---

# 💡 Core Intelligence

WAFER GPT contains three major intelligence layers:

```text
┌─────────────────────────────────────────────────────┐
│                 WAFER GPT INTELLIGENCE              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  01  CNN                                            │
│      WHAT is the predicted wafer pattern?           │
│                                                     │
│  02  Grad-CAM                                       │
│      WHERE did the model focus?                    │
│                                                     │
│  03  Gemini                                         │
│      HOW can the result be understood?             │
│                                                     │
└─────────────────────────────────────────────────────┘
```

This creates an end-to-end explainable AI workflow:

```text
IMAGE
  ↓
CNN CLASSIFICATION
  ↓
PREDICTION
  ↓
CONFIDENCE
  ↓
PROBABILITY DISTRIBUTION
  ↓
GRAD-CAM
  ↓
VISUAL EXPLANATION
  ↓
GEMINI ANALYSIS
  ↓
CONTEXTUAL AI CHAT
```

---

# 🌌 System Architecture

<img src="./assets/architecture.svg" width="100%" alt="WAFER GPT Architecture"/>

### High-Level Architecture

```text
                         USER
                          │
                          ▼
                 ┌─────────────────┐
                 │  WEB FRONTEND   │
                 │ HTML/CSS/JS     │
                 └────────┬────────┘
                          │
                          │ HTTPS
                          ▼
                 ┌─────────────────┐
                 │    FASTAPI      │
                 │    BACKEND      │
                 └────────┬────────┘
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
       TensorFlow      Grad-CAM       Gemini
          CNN         Explainability    AI
            │             │             │
            └─────────────┼─────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ ANALYSIS RESULT │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   RESULTS UI    │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  WAFER GPT CHAT │
                 └─────────────────┘
```

---

# 🔄 Complete AI Pipeline

<img src="./assets/pipeline.svg" width="100%" alt="WAFER GPT AI Pipeline"/>

```text
Upload Wafer Image
        ↓
File Validation
        ↓
Image Storage
        ↓
RGB Conversion
        ↓
224 × 224 Resize
        ↓
Pixel Normalization
        ↓
TensorFlow / Keras Model
        ↓
9-Class Classification
        ↓
Prediction + Confidence
        ↓
Complete Probability Distribution
        ↓
Grad-CAM Generation
        ↓
Gemini AI Analysis
        ↓
Results Dashboard
        ↓
Context-Aware WAFER GPT Chat
```

---

# 🧬 Wafer Defect Classification

WAFER GPT currently supports **9 wafer-pattern classes**.

| # | Class |
|---|---|
| 01 | Center |
| 02 | Donut |
| 03 | Edge-Loc |
| 04 | Edge-Ring |
| 05 | Loc |
| 06 | Near-full |
| 07 | Random |
| 08 | Scratch |
| 09 | none |

The model produces a probability for every class.

### Example

```text
Scratch       91.42%
Random         4.31%
Edge-Loc       2.17%
Loc            1.23%
Center         0.42%
...
```

The class with the highest model probability becomes the predicted class.

---

# 🔬 Defect Classes

### 🎯 Center

Pattern primarily concentrated around the center region of the wafer.

### 🍩 Donut

Ring-like defect distribution forming a donut-shaped pattern.

### 📍 Edge-Loc

Localized defect behavior concentrated around the wafer edge.

### ⭕ Edge-Ring

Ring-shaped defect distribution near the wafer boundary.

### 📌 Loc

Localized defect pattern occurring in a particular wafer region.

### 🟠 Near-full

Defect pattern covering a very large portion of the wafer.

### 🎲 Random

Irregular or randomly distributed defect pattern.

### ✏️ Scratch

Linear or elongated defect pattern resembling a scratch.

### ✅ none

No significant defect pattern detected according to the model classification.

---

# 🧠 CNN Inference

The trained TensorFlow/Keras model expects an input tensor with:

```text
224 × 224 × 3
```

The preprocessing pipeline is:

```text
Original Image
      ↓
RGB Conversion
      ↓
Resize
224 × 224
      ↓
NumPy Array
      ↓
float32
      ↓
Normalize
pixel / 255
      ↓
Add Batch Dimension
      ↓
(1, 224, 224, 3)
      ↓
TensorFlow / Keras Model
      ↓
9 Output Probabilities
```

### Preprocessing

Conceptually:

```python
image = image.convert("RGB")
image = image.resize((224, 224))

image = np.array(
    image,
    dtype=np.float32
)

image = image / 255.0

image = np.expand_dims(
    image,
    axis=0
)
```

---

# 📊 Confidence & Probability Analysis

WAFER GPT does not return only the winning class.

It returns the probability distribution across all supported classes.

### Example

```json
{
  "Center": 0.004,
  "Donut": 0.003,
  "Edge-Loc": 0.0217,
  "Edge-Ring": 0.001,
  "Loc": 0.0123,
  "Near-full": 0.001,
  "Random": 0.0431,
  "Scratch": 0.9142,
  "none": 0.0007
}
```

The frontend can visualize these values using probability bars or charts.

This gives more information than a single class label.

---

# 🔥 Explainable AI — Grad-CAM

<img src="./assets/gradcam.svg" width="100%" alt="Grad-CAM Explainable AI"/>

One of the core features of WAFER GPT is **Grad-CAM**.

A CNN can produce a prediction without directly showing which visual region contributed most strongly to that prediction.

Grad-CAM provides a visual explanation of model attention.

---

# 🧩 Grad-CAM Pipeline

```text
             INPUT IMAGE
                  │
                  ▼
          ┌───────────────┐
          │      CNN      │
          └───────┬───────┘
                  │
                  ▼
        TARGET CONVOLUTION
             FEATURES
                  │
                  ▼
          TARGET CLASS
                  │
                  ▼
          GRADIENT TAPE
                  │
                  ▼
       FEATURE IMPORTANCE
                  │
                  ▼
             HEATMAP
                  │
                  ▼
         HEATMAP RESIZE
                  │
                  ▼
       ORIGINAL + HEATMAP
                  │
                  ▼
         VISUAL EXPLANATION
```

---

# 🔬 How Grad-CAM Works

WAFER GPT dynamically searches for a suitable convolutional layer.

The pipeline then:

1. Finds a convolutional feature layer.
2. Creates a gradient model.
3. Performs forward inference.
4. Selects the predicted class output.
5. Calculates gradients using TensorFlow `GradientTape`.
6. Pools gradients across spatial dimensions.
7. Uses the gradients as feature importance weights.
8. Generates the class activation heatmap.
9. Applies ReLU.
10. Normalizes the heatmap.
11. Resizes it to the image dimensions.
12. Applies an OpenCV color map.
13. Overlays the heatmap on the wafer image.
14. Saves the explainability result.

Conceptually:

```text
CNN Feature Maps
       +
Class Gradients
       ↓
Feature Importance
       ↓
Activation Heatmap
       ↓
Resize
       ↓
Color Mapping
       ↓
Overlay
       ↓
Grad-CAM Visualization
```

---

# 👁️ Why Grad-CAM?

Without explainability:

```text
IMAGE
  ↓
CNN
  ↓
Scratch
```

With WAFER GPT:

```text
IMAGE
  ↓
CNN
  ↓
Scratch
  ↓
Grad-CAM
  ↓
Visual regions influencing the prediction
```

This helps make the classification pipeline more transparent.

> Grad-CAM is an interpretability aid and should not be treated as a definitive physical defect localization method.

---

# 🤖 Gemini AI Analysis

<img src="./assets/gemini-flow.svg" width="100%" alt="Gemini AI Flow"/>

The CNN handles the primary image classification.

The Gemini layer adds natural-language intelligence.

The analysis can use information such as:

```text
Prediction
      +
Confidence
      +
Probability Distribution
      +
Wafer Image
```

This information can be converted into a human-readable analysis.

---

# 🧠 AI Analysis Flow

```text
                 CNN
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
   Prediction  Confidence  Probabilities
        │         │         │
        └─────────┼─────────┘
                  │
                  ▼
             Gemini AI
                  │
                  ▼
       Human-Readable Analysis
```

Gemini can help explain:

- Predicted defect class
- Confidence interpretation
- Probability distribution
- Possible pattern meaning
- Potential hypotheses
- Areas that may require engineering verification

---

# 💬 WAFER GPT Chat

<img src="./assets/chatbot-flow.svg" width="100%" alt="WAFER GPT Chat"/>

WAFER GPT includes a contextual AI assistant.

The chatbot is designed to answer questions about the current wafer analysis.

### Example Questions

```text
Why did the model classify this as Scratch?

What does Edge-Ring mean?

Why is the confidence low?

Explain the result in simple terms.

What could potentially cause this pattern?

What are the alternative predictions?
```

---

# 🧠 Context-Aware Chat

The `/chat` endpoint can receive the current wafer context:

```text
Current Prediction
        +
Confidence
        +
Class Probabilities
        +
Previous AI Analysis
        +
User Question
        ↓
      Gemini
        ↓
Contextual Response
```

This allows the chatbot to operate as a wafer-analysis assistant rather than simply acting as a generic chatbot.

---

# 🔌 Backend API

WAFER GPT uses **FastAPI** as its backend framework.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API status |
| `/health` | GET | Health check |
| `/predict` | POST | Wafer image analysis |
| `/chat` | POST | Contextual AI chat |
| `/uploads/{filename}` | GET | Uploaded image |
| `/gradcam/{filename}` | GET | Grad-CAM result |
| `/docs` | GET | Swagger API documentation |
| `/redoc` | GET | ReDoc API documentation |

---

# 📡 `/predict`

Main wafer analysis endpoint.

### Request

```http
POST /predict
Content-Type: multipart/form-data
```

Upload:

```text
file = wafer_image.png
```

### Example Response

```json
{
  "success": true,
  "filename": "abc123.png",
  "prediction": "Scratch",
  "confidence": 91.42,
  "probabilities": {
    "Center": 0.004,
    "Donut": 0.003,
    "Edge-Loc": 0.0217,
    "Edge-Ring": 0.001,
    "Loc": 0.0123,
    "Near-full": 0.001,
    "Random": 0.0431,
    "Scratch": 0.9142,
    "none": 0.0007
  },
  "gradcam_url": "/gradcam/gradcam_abc123.jpg",
  "llm_analysis": "AI generated analysis",
  "llm_enabled": true
}
```

---

# 💬 `/chat`

The `/chat` endpoint allows contextual interaction with WAFER GPT.

### Request

```json
{
  "message": "Why is the confidence high?",
  "prediction": "Scratch",
  "confidence": 91.42,
  "probabilities": {},
  "llm_analysis": "Previous AI analysis"
}
```

### Response

```json
{
  "success": true,
  "response": "AI generated contextual response..."
}
```

---

# ❤️ Health Monitoring

WAFER GPT provides:

```http
GET /health
```

Example:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "gemini": true
}
```

This allows the frontend or deployment environment to check whether the backend and required AI services are available.

---

# 🏗️ Technology Stack

## Frontend

- HTML5
- CSS3
- JavaScript

## Backend

- Python
- FastAPI
- Uvicorn

## Deep Learning

- TensorFlow
- Keras

## Computer Vision

- OpenCV
- Pillow
- NumPy

## Explainable AI

- Grad-CAM
- TensorFlow GradientTape
- CNN Feature Maps

## Generative AI

- Google Gemini

## Deployment

- Vercel
- Railway

## API Documentation

- Swagger UI
- ReDoc

---

# 📦 Project Structure

```text
WAFER-GPT/
│
├── assets/
│   ├── hero.svg
│   ├── architecture.svg
│   ├── pipeline.svg
│   ├── gradcam.svg
│   ├── gemini-flow.svg
│   ├── chatbot-flow.svg
│   ├── deployment.svg
│   └── dashboard.svg
│
├── backend/
│   ├── main.py
│   ├── llm.py
│   ├── requirements.txt
│   ├── wafer_model.keras
│   │
│   ├── uploads/
│   └── results/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── .gitignore
│
└── README.md
```

---

# ⚙️ Local Setup

## 1. Clone Repository

```bash
git clone https://github.com/Harshitcodes154/WAFER-GPT.git
cd WAFER-GPT
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

---

# 🔐 4. Configure Gemini

Create:

```text
backend/.env
```

Add:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

> Never commit your API key to GitHub.

Your `.gitignore` should contain:

```gitignore
.env
.env.*
__pycache__/
*.pyc
.venv/
```

---

# ▶️ 5. Start Backend

```bash
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

ReDoc:

```text
http://127.0.0.1:8000/redoc
```

---

# 🌐 6. Start Frontend

Open:

```text
frontend/index.html
```

For local development, configure:

```javascript
const API_URL = "http://127.0.0.1:8000";
```

For production:

```javascript
const API_URL = "YOUR_DEPLOYED_BACKEND_URL";
```

---

# 🌍 Production Deployment

WAFER GPT uses a split deployment architecture.

```text
                         INTERNET
                            │
                            ▼
                   ┌─────────────────┐
                   │     VERCEL      │
                   │    FRONTEND     │
                   └────────┬────────┘
                            │
                           HTTPS
                            │
                            ▼
                   ┌─────────────────┐
                   │     RAILWAY     │
                   │     FASTAPI     │
                   └────────┬────────┘
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
        TensorFlow       Grad-CAM       Gemini
          Model        Explainability      AI
```

---

# 🚂 Backend — Railway

The FastAPI backend is deployed on Railway.

### Live Backend

https://wafer-gpt-production.up.railway.app

### Health Check

https://wafer-gpt-production.up.railway.app/health

### Swagger

https://wafer-gpt-production.up.railway.app/docs

### ReDoc

https://wafer-gpt-production.up.railway.app/redoc

---

# ▲ Frontend — Vercel

The frontend can be deployed using Vercel.

The frontend's API configuration should point to the Railway backend:

```javascript
const API_URL =
    "https://wafer-gpt-production.up.railway.app";
```

---

# 🖥️ Frontend Workflow

```text
USER
 │
 ▼
UPLOAD IMAGE
 │
 ▼
FRONTEND VALIDATION
 │
 ▼
FORMDATA
 │
 ▼
POST /predict
 │
 ▼
FASTAPI
 │
 ▼
CNN
 │
 ▼
RESULT JSON
 │
 ├──────────────────┐
 ▼                  ▼
PREDICTION        GRAD-CAM
 │                  │
 └────────┬─────────┘
          ▼
     GEMINI ANALYSIS
          │
          ▼
      RESULTS UI
          │
          ▼
       AI CHAT
```

---

# 📸 Supported Image Formats

WAFER GPT supports:

- PNG
- JPG
- JPEG
- WEBP

The frontend can enforce an upload size limit such as:

```text
10 MB
```

The exact frontend limit depends on the implementation deployed with the project.

---

# 🧪 Image Validation

The frontend workflow can perform:

```text
File Type Validation
        ↓
File Size Validation
        ↓
FileReader Preview
        ↓
Image Preview
        ↓
Enable Analyze Button
```

The backend additionally validates the file extension before processing.

---

# 🖼️ Image Storage

Uploaded images receive generated unique filenames.

Example:

```text
uploads/
└── 8e2f19a1c7d4.png
```

Grad-CAM outputs are stored separately:

```text
results/
└── gradcam_8e2f19a1c7d4.jpg
```

---

# 🛡️ Error Handling

The backend includes handling for:

- Unsupported file formats
- Invalid image files
- Model loading errors
- Grad-CAM failures
- Gemini API errors
- Missing AI responses
- Invalid chat requests

API responses are returned as structured JSON where applicable.

---

# ⚠️ Important AI Disclaimer

WAFER GPT is an **AI-assisted semiconductor analysis system**.

A model prediction should not automatically be considered a confirmed engineering diagnosis.

Model confidence represents the model's output probability and should not be interpreted as absolute certainty.

AI-generated root-cause explanations are hypotheses and should be validated against:

- Process data
- Equipment data
- Manufacturing history
- Physical inspection
- Engineering measurements
- Production context

WAFER GPT is intended as an AI assistance and analysis tool, not a replacement for semiconductor engineering validation.

---

# 🧠 Why Explainable AI Matters

### Traditional AI

```text
IMAGE
  ↓
MODEL
  ↓
CLASS
```

### WAFER GPT

```text
IMAGE
  ↓
MODEL
  ↓
CLASS
  ↓
CONFIDENCE
  ↓
PROBABILITIES
  ↓
GRAD-CAM
  ↓
VISUAL EXPLANATION
  ↓
GEMINI
  ↓
HUMAN-READABLE INSIGHT
  ↓
CONTEXTUAL CHAT
```

The goal is to move from:

> **Prediction**

towards:

> **Prediction + Explanation + Interaction**

---

# ⚔️ Traditional Classifier vs WAFER GPT

| Capability | Traditional Classifier | WAFER GPT |
|---|---:|---:|
| Image Classification | ✅ | ✅ |
| Confidence | Sometimes | ✅ |
| Class Probabilities | Sometimes | ✅ |
| Explainability | Limited | ✅ |
| Grad-CAM | ❌ | ✅ |
| Natural Language Analysis | ❌ | ✅ |
| Contextual Chat | ❌ | ✅ |
| REST API | Depends | ✅ |
| Interactive Web UI | Depends | ✅ |
| Cloud Deployment | Depends | ✅ |

---

# 🎯 Real-World Potential

WAFER GPT can serve as a foundation for AI-assisted semiconductor workflows:

```text
SEMICONDUCTOR INSPECTION
          ↓
WAFER PATTERN ANALYSIS
          ↓
DEFECT CLASSIFICATION
          ↓
EXPLAINABLE AI
          ↓
AI-ASSISTED INTERPRETATION
          ↓
ENGINEERING INVESTIGATION
```

Potential applications include:

- Automated wafer inspection
- Semiconductor defect classification
- Visual quality analysis
- Explainable manufacturing AI
- AI-assisted engineering workflows
- Research and experimentation
- Wafer pattern analytics
- Intelligent inspection interfaces

---

# 🚀 Future Roadmap

## 🔹 Phase 1 — Current

- [x] Wafer image upload
- [x] Image preprocessing
- [x] CNN classification
- [x] 9-class prediction
- [x] Confidence score
- [x] Probability distribution
- [x] Grad-CAM
- [x] Gemini AI analysis
- [x] Context-aware chat
- [x] FastAPI backend
- [x] Interactive frontend
- [x] Cloud deployment

---

## 🔹 Phase 2 — Planned

```text
Multiple Images
      ↓
Batch Analysis
      ↓
Wafer Comparison
      ↓
Historical Analysis
      ↓
Defect Trends
      ↓
Automated Reports
```

Potential features:

- Multi-image batch analysis
- Wafer-to-wafer comparison
- Historical analysis
- Defect trend dashboard
- Automated PDF reports
- Exportable analysis
- Advanced defect localization
- Model monitoring

---

## 🔹 Phase 3 — Advanced

```text
Multiple Wafers
      ↓
Defect Database
      ↓
Historical Trends
      ↓
Process Correlation
      ↓
Yield Analytics
      ↓
Predictive Intelligence
      ↓
Engineering Copilot
```

Potential future capabilities:

- Yield prediction
- Process correlation
- Historical defect intelligence
- Equipment-level analytics
- Engineer feedback loops
- Continuous model improvement
- Automated root-cause investigation
- Production-scale monitoring

---

# 🔮 Vision

The long-term vision of WAFER GPT is to move beyond:

```text
AI THAT CLASSIFIES WAFERS
```

towards:

```text
AI THAT HELPS ENGINEERS
UNDERSTAND WAFER BEHAVIOR
```

The combination of:

```text
Computer Vision
      +
Deep Learning
      +
Explainable AI
      +
Generative AI
      +
Interactive Assistance
```

creates a foundation for intelligent semiconductor inspection systems.

---

# 🏆 Project Highlights

```text
✓ 9-Class Wafer Classification
✓ TensorFlow / Keras Deep Learning
✓ Real-Time Inference
✓ Probability Distribution
✓ Grad-CAM Explainability
✓ Gemini AI Analysis
✓ Context-Aware AI Chat
✓ FastAPI REST Backend
✓ Interactive Web Interface
✓ Swagger / ReDoc API
✓ Cloud Deployment
✓ Vercel + Railway Architecture
```

---

# 📊 Technical Specifications

| Component | Technology |
|---|---|
| Frontend | HTML / CSS / JavaScript |
| Backend | FastAPI |
| Server | Uvicorn |
| ML Framework | TensorFlow / Keras |
| Image Processing | OpenCV / Pillow |
| Numerical Processing | NumPy |
| Explainability | Grad-CAM |
| Generative AI | Google Gemini |
| Frontend Deployment | Vercel |
| Backend Deployment | Railway |
| API Documentation | Swagger / ReDoc |

---

# 🔐 Security Notes

Never expose:

```text
GEMINI_API_KEY
```

in:

- GitHub repositories
- Frontend JavaScript
- README files
- Screenshots
- Public logs
- Client-side code

The Gemini API key should remain **server-side**.

Recommended `.gitignore`:

```gitignore
.env
.env.*
__pycache__/
*.pyc
.venv/
```

---

# 🧪 Example End-to-End Analysis

```text
STEP 01
User uploads wafer image
          ↓

STEP 02
Frontend validates image
          ↓

STEP 03
FastAPI receives image
          ↓

STEP 04
Image converted to RGB
          ↓

STEP 05
Image resized to 224 × 224
          ↓

STEP 06
Pixels normalized
          ↓

STEP 07
TensorFlow model performs inference
          ↓

STEP 08
9 probabilities generated
          ↓

STEP 09
Highest probability becomes prediction
          ↓

STEP 10
Grad-CAM generated
          ↓

STEP 11
Gemini receives analysis context
          ↓

STEP 12
Human-readable explanation generated
          ↓

STEP 13
Results displayed in dashboard
          ↓

STEP 14
User asks WAFER GPT follow-up questions
```

---

# 🌐 Live Project

## 🚀 Backend

**Railway**

https://wafer-gpt-production.up.railway.app

## 📚 API Documentation

https://wafer-gpt-production.up.railway.app/docs

## 🩺 Health Check

https://wafer-gpt-production.up.railway.app/health

## 💻 Source Code

https://github.com/Harshitcodes154/WAFER-GPT

> Add the Vercel frontend URL here once the public frontend deployment is finalized.

---

# 🖼️ Project Visuals

## Dashboard

<img src="./assets/dashboard.svg" width="100%" alt="WAFER GPT Dashboard"/>

---

## Architecture

<img src="./assets/architecture.svg" width="100%" alt="WAFER GPT Architecture"/>

---

## AI Pipeline

<img src="./assets/pipeline.svg" width="100%" alt="WAFER GPT Pipeline"/>

---

## Grad-CAM

<img src="./assets/gradcam.svg" width="100%" alt="WAFER GPT Grad-CAM"/>

---

## Gemini AI Flow

<img src="./assets/gemini-flow.svg" width="100%" alt="Gemini AI Flow"/>

---

## Chatbot Flow

<img src="./assets/chatbot-flow.svg" width="100%" alt="WAFER GPT Chatbot"/>

---

# 💻 API Quick Test

After starting the backend:

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "gemini": true
}
```

---

# 🧰 Development Commands

## Start Backend

```bash
uvicorn main:app --reload
```

## Start on Specific Port

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Git Workflow

```bash
git add .
git commit -m "Update WAFER GPT"
git push
```

---

# 🤝 Contributing

Contributions are welcome.

Possible contribution areas:

```text
Model Improvements
       ↓
UI / UX
       ↓
Explainability
       ↓
AI Analysis
       ↓
API Improvements
       ↓
Deployment
       ↓
Analytics
```

## Contribution Flow

Fork the repository.

Create a branch:

```bash
git checkout -b feature/new-feature
```

Commit your changes:

```bash
git add .
git commit -m "Add new feature"
```

Push:

```bash
git push origin feature/new-feature
```

Then open a Pull Request.

---

# ⭐ Support the Project

If you find WAFER GPT interesting:

⭐ Star the repository

🍴 Fork the repository

🐛 Report bugs

💡 Suggest improvements

🤝 Contribute

---

# 👨‍💻 Author

<div align="center">

## Harshit Kumar

### AI/ML Developer • Hackathon Builder • Full-Stack AI Developer

<br/>

<a href="https://github.com/Harshitcodes154">
<img src="https://img.shields.io/badge/GitHub-Harshitcodes154-181717?style=for-the-badge&logo=github" alt="GitHub"/>
</a>

<br/><br/>

Built with ❤️ using

**TensorFlow • FastAPI • OpenCV • Grad-CAM • Gemini AI**

</div>

---

<div align="center">

# ⚡ WAFER GPT

### DETECT. EXPLAIN. UNDERSTAND.

<img src="./assets/hero.svg" width="100%" alt="WAFER GPT"/>

<br/><br/>

**AI-Powered Semiconductor Wafer Defect Intelligence**

<br/>

⭐ Star the repository if you find the project useful.

</div>
