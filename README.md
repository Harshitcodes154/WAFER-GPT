<div align="center">

# ⚡ WAFER GPT

### AI-Powered Semiconductor Wafer Defect Intelligence

<img src="./assets/hero.svg" width="100%" alt="WAFER GPT"/>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=18&pause=1000&color=22D3EE&center=true&vCenter=true&width=850&lines=DETECT+%7C+CLASSIFY+%7C+EXPLAIN+%7C+UNDERSTAND;AI+POWERED+WAFER+DEFECT+INTELLIGENCE;COMPUTER+VISION+%2B+EXPLAINABLE+AI+%2B+GENERATIVE+AI;TURNING+WAFER+IMAGES+INTO+ACTIONABLE+INSIGHTS"/>

<br/><br/>

<a href="https://github.com/Harshitcodes154/WAFER-GPT">
<img src="https://img.shields.io/badge/GitHub-WAFER--GPT-181717?style=for-the-badge&logo=github"/>
</a>

<img src="https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>

<img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>

<img src="https://img.shields.io/badge/Google%20Gemini-Generative%20AI-4285F4?style=for-the-badge&logo=google&logoColor=white"/>

<img src="https://img.shields.io/badge/Grad--CAM-Explainable%20AI-7C3AED?style=for-the-badge"/>

<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>

<br/><br/>

### 🚀 Detect. Explain. Understand.

**WAFER GPT is an AI-powered semiconductor wafer defect intelligence platform that combines Deep Learning, Computer Vision, Grad-CAM Explainable AI and Google Gemini to analyze wafer patterns and generate human-readable insights.**

</div>

---

# 🧠 What is WAFER GPT?

**WAFER GPT** is an end-to-end AI system designed for intelligent semiconductor wafer defect analysis.

Instead of providing only a machine-learning classification, WAFER GPT creates a complete analysis pipeline:

```text
                ┌──────────────────────┐
                │     WAFER IMAGE      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ IMAGE PREPROCESSING  │
                │   PIL + NumPy + CV   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   TENSORFLOW / CNN   │
                │   DEFECT CLASSIFIER  │
                └──────────┬───────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
        PREDICTION     CONFIDENCE   PROBABILITIES
             │
             ▼
        ┌───────────────┐
        │   GRAD-CAM    │
        │ EXPLANABILITY │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │   GEMINI AI   │
        │  EXPLANATION  │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │ WAFER GPT CHAT│
        │   ASSISTANT   │
        └───────────────┘
