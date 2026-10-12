
import { Client, handle_file } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

// ============================================================
// WAFER-GPT FRONTEND — HUGGING FACE INTEGRATION
// ============================================================

const SPACE_ID = "harshitcodes1544/WAFER-GPT";

// Confirm this endpoint in your Space's "Use via API" section.
// If the API page shows another name, update this value.
const PREDICT_ENDPOINT = "/predict_wafer";

let selectedFile = null;
let currentAnalysis = null;
let gradioClientPromise = null;
let isAnalyzing = false;

// ============================================================
// DOM ELEMENTS
// ============================================================

const $ = (id) => document.getElementById(id);

const uploadBox = $("uploadBox");
const fileInput = $("fileInput");
const browseText = $("browseText");
const uploadPlaceholder = $("uploadPlaceholder");
const previewContainer = $("previewContainer");
const previewImage = $("previewImage");
const fileName = $("fileName");
const fileSize = $("fileSize");
const removeBtn = $("removeBtn");

const analyzeBtn = $("analyzeBtn");
const loadingSection = $("loadingSection");
const resultsSection = $("resultsSection");
const newAnalysisBtn = $("newAnalysisBtn");

const prediction = $("prediction");
const confidence = $("confidence");
const confidenceBar = $("confidenceBar");
const severity = $("severity");
const originalImage = $("originalImage");
const gradcamImage = $("gradcamImage");
const probabilities = $("probabilities");
const llmAnalysis = $("llmAnalysis");

const chatForm = $("chatForm");
const chatInput = $("chatInput");
const chatSendBtn = $("chatSendBtn");
const chatMessages = $("chatMessages");
const chatStatus = $("chatStatus");

console.log("WAFER-GPT frontend loaded.");

// ============================================================
// HUGGING FACE GRADIO CLIENT
// ============================================================

async function getGradioClient() {
    if (!gradioClientPromise) {
        gradioClientPromise = Client.connect(SPACE_ID);
    }

    try {
        return await gradioClientPromise;
    } catch (error) {
        gradioClientPromise = null;
        throw error;
    }
}

// ============================================================
// FILE VALIDATION
// ============================================================

function validateFile(file) {
    if (!file) {
        alert("Please select a wafer image first.");
        return false;
    }

    const allowedTypes = [
        "image/png",
        "image/jpeg",
        "image/webp"
    ];

    if (!allowedTypes.includes(file.type)) {
        alert("Please select a PNG, JPG, JPEG or WEBP image.");
        return false;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert("Maximum image size is 10 MB.");
        return false;
    }

    return true;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;

    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// ============================================================
// IMAGE PREVIEW
// ============================================================

function showPreview(file) {
    if (!validateFile(file)) {
        return;
    }

    selectedFile = file;

    if (previewImage) {
        previewImage.src = URL.createObjectURL(file);
    }

    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = formatFileSize(file.size);

    if (uploadPlaceholder) {
        uploadPlaceholder.style.display = "none";
    }

    if (previewContainer) {
        previewContainer.style.display = "block";
    }

    if (analyzeBtn) {
        analyzeBtn.disabled = false;
        analyzeBtn.style.opacity = "1";
        analyzeBtn.style.cursor = "pointer";
    }

    // Do not automatically run inference when previewing.
}

if (fileInput) {
    fileInput.addEventListener("change", () => {
        const file = fileInput.files?.[0];
        if (file) showPreview(file);
    });
}

if (browseText && fileInput) {
    browseText.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        fileInput.click();
    });
}

if (uploadBox && fileInput) {
    uploadBox.addEventListener("click", (event) => {
        if (
            event.target === removeBtn ||
            removeBtn?.contains(event.target) ||
            previewContainer?.contains(event.target)
        ) {
            return;
        }

        event.preventDefault();
        fileInput.click();
    });

    uploadBox.addEventListener("dragover", (event) => {
        event.preventDefault();
        uploadBox.classList.add("dragging");
    });

    uploadBox.addEventListener("dragleave", () => {
        uploadBox.classList.remove("dragging");
    });

    uploadBox.addEventListener("drop", (event) => {
        event.preventDefault();
        uploadBox.classList.remove("dragging");

        const file = event.dataTransfer?.files?.[0];
        if (file) showPreview(file);
    });
}

// ============================================================
// RESET UPLOAD
// ============================================================

function resetUpload() {
    selectedFile = null;
    currentAnalysis = null;

    if (fileInput) fileInput.value = "";

    if (previewImage) {
        if (previewImage.src.startsWith("blob:")) {
            URL.revokeObjectURL(previewImage.src);
        }
        previewImage.removeAttribute("src");
    }

    if (uploadPlaceholder) {
        uploadPlaceholder.style.display = "block";
    }

    if (previewContainer) {
        previewContainer.style.display = "none";
    }

    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.style.opacity = "0.6";
        analyzeBtn.style.cursor = "not-allowed";
    }
}

if (removeBtn) {
    removeBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        resetUpload();
    });
}

// ============================================================
// ANALYZE BUTTON
// ============================================================

if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async (event) => {
        event.preventDefault();

        if (!selectedFile) {
            alert("Please upload a wafer image first.");
            return;
        }

        await analyzeWafer();
    });
}

// ============================================================
// RUN MODEL ON HUGGING FACE
// ============================================================

async function analyzeWafer() {
    if (!selectedFile || isAnalyzing) return;

    isAnalyzing = true;

    if (loadingSection) loadingSection.style.display = "block";
    if (resultsSection) resultsSection.style.display = "none";

    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML =
            "<span>Analyzing...</span><span>⏳</span>";
    }

    try {
        console.log("Connecting to Hugging Face Space:", SPACE_ID);

        const client = await getGradioClient();

        console.log("Uploading image and requesting prediction...");

        const result = await client.predict(PREDICT_ENDPOINT, [
            await handle_file(selectedFile)
        ]);

        console.log("Gradio API response:", result);

        const normalized = normalizeResult(result);
        displayResults(normalized);

    } catch (error) {
        console.error("Wafer analysis failed:", error);

        const message = error?.message || String(error);

        alert(
            "Analysis failed.\n\n" +
            "Check these things:\n" +
            "1. Your Hugging Face Space is running.\n" +
            "2. The API endpoint is correct.\n" +
            "3. The Space's prediction function accepts an image.\n\n" +
            "Error: " + message
        );

    } finally {
        isAnalyzing = false;

        if (loadingSection) {
            loadingSection.style.display = "none";
        }

        if (analyzeBtn) {
            analyzeBtn.disabled = !selectedFile;
            analyzeBtn.innerHTML =
                "<span>Analyze Wafer</span><span>→</span>";
        }
    }
}

// ============================================================
// NORMALIZE MODEL RESPONSE
// ============================================================

function normalizeResult(result) {
    const data = result?.data;

    if (!Array.isArray(data) || data.length === 0) {
        throw new Error(
            "The Space returned an empty response. Check its output format."
        );
    }

    // Gradio may return a dictionary, a label, or a list of labels/scores.
    let raw = data[0];

    // Some interfaces return a prediction dictionary in the first output.
    if (raw && typeof raw === "object" && !Array.isArray(raw)) {
        const scores =
            raw.probabilities ??
            raw.confidences ??
            raw.scores ??
            null;

        if (scores && typeof scores === "object") {
            const entries = Array.isArray(scores)
                ? scores.map((value, index) => [String(index), value])
                : Object.entries(scores);

            if (entries.length > 0) {
                const best = entries.reduce((a, b) =>
                    Number(b[1]) > Number(a[1]) ? b : a
                );

                const maxValue = Number(best[1]);
                const usesPercent = entries.some(
                    ([, value]) => Number(value) > 1
                );

                const probabilityMap = {};

                for (const [label, value] of entries) {
                    let number = Number(value);
                    if (!Number.isFinite(number)) continue;

                    if (usesPercent) number /= 100;
                    probabilityMap[label] = number;
                }

                let predictedLabel =
                    raw.prediction ?? raw.label ?? best[0];

                let conf = Number(raw.confidence);

                if (!Number.isFinite(conf)) {
                    conf = usesPercent
                        ? maxValue
                        : maxValue * 100;
                } else if (conf <= 1) {
                    conf *= 100;
                }

                return {
                    prediction: String(predictedLabel),
                    confidence: conf,
                    probabilities: probabilityMap,
                    llm_analysis:
                        raw.llm_analysis ??
                        raw.analysis ??
                        ""
                };
            }
        }

        // Handle an object that contains only a predicted class.
        if (raw.prediction || raw.label) {
            let conf = Number(raw.confidence ?? 0);

            if (conf > 0 && conf <= 1) conf *= 100;

            return {
                prediction: String(raw.prediction ?? raw.label),
                confidence: conf,
                probabilities: {},
                llm_analysis:
                    raw.llm_analysis ?? raw.analysis ?? ""
            };
        }
    }

    // A plain string output may be the predicted class.
    if (typeof raw === "string") {
        return {
            prediction: raw,
            confidence: 0,
            probabilities: {},
            llm_analysis: ""
        };
    }

    // Support an array of {label, confidence/score} entries.
    if (Array.isArray(raw)) {
        const entries = raw.filter(
            item =>
                item &&
                typeof item === "object" &&
                ("label" in item)
        );

        if (entries.length) {
            const scoreOf = item =>
                Number(item.confidence ?? item.score ?? 0);

            const best = entries.reduce((a, b) =>
                scoreOf(b) > scoreOf(a) ? b : a
            );

            const probabilityMap = {};

            for (const item of entries) {
                let score = scoreOf(item);
                if (score > 1) score /= 100;
                probabilityMap[String(item.label)] = score;
            }

            let conf = scoreOf(best);
            if (conf <= 1) conf *= 100;

            return {
                prediction: String(best.label),
                confidence: conf,
                probabilities: probabilityMap,
                llm_analysis: ""
            };
        }
    }

    throw new Error(
        "Unsupported model response. Open the browser console to inspect the Gradio response."
    );
}

// ============================================================
// DISPLAY RESULTS
// ============================================================

function displayResults(data) {
    currentAnalysis = data;

    if (prediction) {
        prediction.textContent = data.prediction || "Unknown";
    }

    const conf = Math.max(
        0,
        Math.min(100, Number(data.confidence) || 0)
    );

    if (confidence) {
        confidence.textContent = `${conf.toFixed(2)}%`;
    }

    if (confidenceBar) {
        confidenceBar.style.width = `${conf}%`;
    }

    if (severity) {
        severity.textContent = calculateSeverity(conf);
    }

    // Show the uploaded image as the original image.
    if (originalImage && selectedFile) {
        if (originalImage.src.startsWith("blob:")) {
            URL.revokeObjectURL(originalImage.src);
        }

        originalImage.src = URL.createObjectURL(selectedFile);
    }

    // The current classification function may not return Grad-CAM.
    // Clear the image instead of showing a broken URL.
    if (gradcamImage) {
        gradcamImage.removeAttribute("src");
    }

    renderProbabilities(data.probabilities || {});

    if (llmAnalysis) {
        llmAnalysis.textContent =
            data.llm_analysis ||
            "AI analysis is not included in the current model response.";
    }

    if (chatInput) chatInput.disabled = false;
    if (chatSendBtn) chatSendBtn.disabled = false;

    if (chatStatus) {
        chatStatus.textContent =
            "Prediction ready. Chat requires a configured chat backend.";
    }

    if (resultsSection) {
        resultsSection.style.display = "block";
        resultsSection.style.visibility = "visible";
        resultsSection.style.opacity = "1";

        setTimeout(() => {
            resultsSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });
        }, 200);
    }
}

// ============================================================
// SEVERITY
// ============================================================

function calculateSeverity(value) {
    if (value >= 90) return "HIGH";
    if (value >= 70) return "MEDIUM";
    return "LOW";
}

// ============================================================
// PROBABILITY BARS
// ============================================================

function renderProbabilities(data) {
    if (!probabilities) return;

    probabilities.innerHTML = "";

    const entries = Object.entries(data).sort(
        (a, b) => Number(b[1]) - Number(a[1])
    );

    for (const [label, value] of entries) {
        let percentage = Number(value) * 100;

        if (Number(value) > 1) {
            percentage = Number(value);
        }

        percentage = Math.max(0, Math.min(100, percentage));

        const row = document.createElement("div");
        row.className = "probability-row";

        const info = document.createElement("div");
        info.className = "probability-info";

        const labelElement = document.createElement("span");
        labelElement.textContent = label;

        const valueElement = document.createElement("span");
        valueElement.textContent = `${percentage.toFixed(2)}%`;

        info.append(labelElement, valueElement);

        const track = document.createElement("div");
        track.className = "probability-track";

        const fill = document.createElement("div");
        fill.className = "probability-fill";
        fill.style.width = `${percentage}%`;

        track.appendChild(fill);
        row.append(info, track);
        probabilities.appendChild(row);
    }
}

// ============================================================
// NEW ANALYSIS
// ============================================================

if (newAnalysisBtn) {
    newAnalysisBtn.addEventListener("click", event => {
        event.preventDefault();

        if (resultsSection) {
            resultsSection.style.display = "none";
        }

        resetUpload();

        if (chatInput) {
            chatInput.value = "";
            chatInput.disabled = true;
        }

        if (chatSendBtn) chatSendBtn.disabled = true;

        if (chatStatus) {
            chatStatus.textContent =
                "Analyze a wafer to enable the assistant.";
        }

        if (chatMessages) {
            chatMessages.innerHTML = "";

            const message = document.createElement("div");
            message.className =
                "chat-message assistant-message";
            message.textContent =
                "Upload and analyze a wafer, then ask about its prediction, confidence, or defect pattern.";

            chatMessages.appendChild(message);
        }

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    });
}

// ============================================================
// CHAT UI
// ============================================================

function addChatMessage(message, role) {
    if (!chatMessages) return;

    const element = document.createElement("div");
    element.className = `chat-message ${role}-message`;
    element.textContent = message;

    chatMessages.appendChild(element);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

if (chatForm) {
    chatForm.addEventListener("submit", event => {
        event.preventDefault();

        const message = chatInput?.value.trim();

        if (!message || !currentAnalysis) return;

        addChatMessage(message, "user");

        if (chatInput) chatInput.value = "";

        addChatMessage(
            "The current Hugging Face classification integration does not expose a chat endpoint. A backend /chat function must be configured to enable AI replies.",
            "assistant"
        );

        if (chatStatus) {
            chatStatus.textContent =
                "Chat backend is not configured.";
        }
    });
}

// ============================================================
// INITIAL STATE
// ============================================================

if (loadingSection) {
    loadingSection.style.display = "none";
}

if (resultsSection) {
    resultsSection.style.display = "none";
}

if (analyzeBtn) {
    analyzeBtn.disabled = true;
}

if (chatInput) chatInput.disabled = true;
if (chatSendBtn) chatSendBtn.disabled = true;

if (chatStatus) {
    chatStatus.textContent =
        "Analyze a wafer to enable the assistant.";
}

console.log("WAFER-GPT is ready.");
