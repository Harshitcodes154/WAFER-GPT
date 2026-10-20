// ============================================================
// CONFIGURATION & DOM ELEMENTS
// ============================================================

// Vercel deployment par frontend aur backend same domain par hain, isliye relative URL
const API_BASE = "";

let selectedFile = null;
let currentAnalysis = null;
let isAnalyzing = false;

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

// ============================================================
// FILE UPLOAD AND PREVIEW
// ============================================================

function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function showPreview(file) {
    if (!file) return;

    if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
        alert("Please select a PNG, JPG, JPEG or WEBP image.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert("Maximum image size is 10 MB.");
        return;
    }

    selectedFile = file;

    if (previewImage) {
        if (previewImage.src.startsWith("blob:")) {
            URL.revokeObjectURL(previewImage.src);
        }
        previewImage.src = URL.createObjectURL(file);
    }

    if (fileName) fileName.textContent = file.name;
    if (fileSize) fileSize.textContent = formatFileSize(file.size);

    if (uploadPlaceholder) uploadPlaceholder.style.display = "none";
    if (previewContainer) previewContainer.style.display = "block";

    if (analyzeBtn) {
        analyzeBtn.disabled = false;
        analyzeBtn.style.opacity = "1";
        analyzeBtn.style.cursor = "pointer";
    }
}

if (fileInput) {
    fileInput.addEventListener("change", () => {
        showPreview(fileInput.files?.[0]);
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
        showPreview(event.dataTransfer?.files?.[0]);
    });
}

// ============================================================
// RESET
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

    if (uploadPlaceholder) uploadPlaceholder.style.display = "block";
    if (previewContainer) previewContainer.style.display = "none";

    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.style.opacity = "0.6";
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
// ANALYZE IMAGE (DIRECT FASTAPI /predict CALL)
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

async function analyzeWafer() {
    if (!selectedFile || isAnalyzing) return;

    isAnalyzing = true;

    if (loadingSection) loadingSection.style.display = "block";
    if (resultsSection) resultsSection.style.display = "none";

    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = "<span>Analyzing...</span><span>⏳</span>";
    }

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || `Server responded with ${response.status}`);
        }

        const result = await response.json();
        console.log("Prediction response:", result);

        if (!result.success) {
            throw new Error(result.detail || "Prediction request was unsuccessful.");
        }

        currentAnalysis = {
            prediction: result.prediction,
            confidence: result.confidence,
            probabilities: result.probabilities,
            llm_analysis: result.llm_analysis || "",
        };

        displayResults(currentAnalysis, result.gradcam_url);

    } catch (error) {
        console.error("Analysis failed:", error);
        alert(
            "Analysis failed.\n\n" +
            (error?.message || String(error)) +
            "\n\nCheck browser console and Vercel function runtime logs."
        );
    } finally {
        isAnalyzing = false;

        if (loadingSection) loadingSection.style.display = "none";

        if (analyzeBtn) {
            analyzeBtn.disabled = !selectedFile;
            analyzeBtn.innerHTML = "<span>Analyze Wafer</span><span>→</span>";
        }
    }
}

// ============================================================
// SHOW RESULTS AND HEATMAP
// ============================================================

function displayResults(data, heatmapUrl) {
    if (prediction) prediction.textContent = data.prediction;

    const conf = Math.max(0, Math.min(100, data.confidence || 0));

    if (confidence) confidence.textContent = `${conf.toFixed(2)}%`;
    if (confidenceBar) confidenceBar.style.width = `${conf}%`;

    if (severity) {
        severity.textContent =
            conf >= 90 ? "HIGH" : conf >= 70 ? "MEDIUM" : "LOW";
    }

    if (originalImage && selectedFile) {
        if (originalImage.src.startsWith("blob:")) {
            URL.revokeObjectURL(originalImage.src);
        }
        originalImage.src = URL.createObjectURL(selectedFile);
    }

    if (gradcamImage) {
        if (heatmapUrl) {
            // Cache bust parameter ensures the newest generated heatmap loads
            gradcamImage.src = `${API_BASE}${heatmapUrl}?t=${Date.now()}`;
            gradcamImage.style.display = "block";
        } else {
            gradcamImage.removeAttribute("src");
            gradcamImage.style.display = "none";
            console.warn("No Heatmap URL returned by the backend.");
        }
    }

    renderProbabilities(data.probabilities);

    if (llmAnalysis) {
        llmAnalysis.textContent =
            data.llm_analysis ||
            "Ask the Gemini assistant below for an explanation of this prediction.";
    }

    if (chatInput) chatInput.disabled = false;
    if (chatSendBtn) chatSendBtn.disabled = false;

    if (chatStatus) {
        chatStatus.textContent = "Gemini assistant is ready.";
    }

    if (resultsSection) {
        resultsSection.style.display = "block";
        resultsSection.style.visibility = "visible";
        resultsSection.style.opacity = "1";

        setTimeout(() => {
            resultsSection.scrollIntoView({
                behavior: "smooth",
                block: "start",
            });
        }, 200);
    }
}

// ============================================================
// CLASS PROBABILITIES
// ============================================================

function renderProbabilities(data) {
    if (!probabilities) return;

    probabilities.innerHTML = "";

    const entries = Object.entries(data || {}).sort(
        (a, b) => Number(b[1]) - Number(a[1])
    );

    for (const [label, value] of entries) {
        const percentage = Math.max(
            0,
            Math.min(100, Number(value) * 100)
        );

        const row = document.createElement("div");
        row.className = "probability-row";

        const info = document.createElement("div");
        info.className = "probability-info";

        const labelEl = document.createElement("span");
        labelEl.textContent = label;

        const valueEl = document.createElement("span");
        valueEl.textContent = `${percentage.toFixed(2)}%`;

        info.append(labelEl, valueEl);

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
// GEMINI CHAT (DIRECT FASTAPI /chat CALL)
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
    chatForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        const message = chatInput?.value.trim();

        if (!message) return;

        if (!currentAnalysis) {
            alert("Please analyze a wafer before asking Gemini.");
            return;
        }

        addChatMessage(message, "user");

        if (chatInput) {
            chatInput.value = "";
            chatInput.disabled = true;
        }

        if (chatSendBtn) chatSendBtn.disabled = true;
        if (chatStatus) chatStatus.textContent = "Gemini is thinking...";

        try {
            const payload = {
                message: message,
                prediction: currentAnalysis.prediction,
                confidence: currentAnalysis.confidence,
                probabilities: currentAnalysis.probabilities,
                llm_analysis: currentAnalysis.llm_analysis || "",
            };

            const response = await fetch(`${API_BASE}/chat`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server responded with ${response.status}`);
            }

            const data = await response.json();
            const answer = data?.response;

            if (typeof answer !== "string" || !answer.trim()) {
                throw new Error("Gemini returned an empty response.");
            }

            addChatMessage(answer, "assistant");

            if (chatStatus) {
                chatStatus.textContent = "Gemini assistant is ready.";
            }

        } catch (error) {
            console.error("Gemini chat failed:", error);

            addChatMessage(
                "Gemini could not respond: " +
                (error?.message || String(error)),
                "assistant"
            );

            if (chatStatus) {
                chatStatus.textContent = "Chat failed. Verify GEMINI_API_KEY in Vercel settings.";
            }

        } finally {
            if (chatInput) {
                chatInput.disabled = false;
                chatInput.focus();
            }

            if (chatSendBtn) chatSendBtn.disabled = false;
        }
    });
}

// ============================================================
// NEW ANALYSIS
// ============================================================

if (newAnalysisBtn) {
    newAnalysisBtn.addEventListener("click", (event) => {
        event.preventDefault();

        if (resultsSection) resultsSection.style.display = "none";

        resetUpload();

        if (chatInput) {
            chatInput.value = "";
            chatInput.disabled = true;
        }

        if (chatSendBtn) chatSendBtn.disabled = true;

        if (chatStatus) {
            chatStatus.textContent = "Analyze a wafer to enable the assistant.";
        }

        if (chatMessages) {
            chatMessages.innerHTML = "";

            const welcome = document.createElement("div");
            welcome.className = "chat-message assistant-message";
            welcome.textContent =
                "Upload and analyze a wafer, then ask me about the prediction.";

            chatMessages.appendChild(welcome);
        }

        window.scrollTo({ top: 0, behavior: "smooth" });
    });
}

// ============================================================
// INITIAL STATE
// ============================================================

if (loadingSection) loadingSection.style.display = "none";
if (resultsSection) resultsSection.style.display = "none";
if (analyzeBtn) analyzeBtn.disabled = true;
if (chatInput) chatInput.disabled = true;
if (chatSendBtn) chatSendBtn.disabled = true;

if (chatStatus) {
    chatStatus.textContent = "Analyze a wafer to enable the Gemini assistant.";
}

console.log("WAFER-GPT frontend ready.");
