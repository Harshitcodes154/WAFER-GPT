// ============================================================
// WAFER GPT FRONTEND
// ============================================================

const API_URL = "http://127.0.0.1:8001";

let selectedFile = null;


// ============================================================
// DOM
// ============================================================

const uploadBox =
    document.getElementById("uploadBox");

const fileInput =
    document.getElementById("fileInput");

const browseText =
    document.getElementById("browseText");

const uploadPlaceholder =
    document.getElementById("uploadPlaceholder");

const previewContainer =
    document.getElementById("previewContainer");

const previewImage =
    document.getElementById("previewImage");

const fileName =
    document.getElementById("fileName");

const fileSize =
    document.getElementById("fileSize");

const removeBtn =
    document.getElementById("removeBtn");

const analyzeBtn =
    document.getElementById("analyzeBtn");

const loadingSection =
    document.getElementById("loadingSection");

const resultsSection =
    document.getElementById("resultsSection");

const prediction =
    document.getElementById("prediction");

const confidence =
    document.getElementById("confidence");

const confidenceBar =
    document.getElementById("confidenceBar");

const severity =
    document.getElementById("severity");

const originalImage =
    document.getElementById("originalImage");

const gradcamImage =
    document.getElementById("gradcamImage");

const probabilities =
    document.getElementById("probabilities");

const llmAnalysis =
    document.getElementById("llmAnalysis");

const newAnalysisBtn =
    document.getElementById("newAnalysisBtn");


// ============================================================
// START
// ============================================================

console.log("====================================");
console.log("WAFER GPT FRONTEND LOADED");
console.log("====================================");

console.log(
    "uploadBox:",
    uploadBox
);

console.log(
    "fileInput:",
    fileInput
);

console.log(
    "analyzeBtn:",
    analyzeBtn
);


// ============================================================
// BROWSE
// ============================================================

browseText.addEventListener(
    "click",
    function (event) {

        event.preventDefault();

        event.stopPropagation();

        console.log(
            "BROWSE CLICKED"
        );

        fileInput.click();

    }
);


// ============================================================
// UPLOAD BOX
// ============================================================

uploadBox.addEventListener(
    "click",
    function (event) {

        event.preventDefault();

        event.stopPropagation();


        if (
            event.target === removeBtn ||
            removeBtn.contains(event.target)
        ) {

            return;

        }


        if (
            previewContainer.style.display ===
            "block"
        ) {

            return;

        }


        console.log(
            "UPLOAD BOX CLICKED"
        );


        fileInput.click();

    }
);


// ============================================================
// FILE CHANGE
// ============================================================

fileInput.addEventListener(
    "change",
    function () {

        console.log(
            "FILE INPUT CHANGED"
        );


        const file =
            fileInput.files[0];


        if (!file) {

            console.log(
                "NO FILE SELECTED"
            );

            return;
        }


        console.log(
            "Selected file:",
            file.name
        );


        selectedFile =
            file;


        showPreview(file);

    }
);


// ============================================================
// SHOW PREVIEW
// ============================================================

function showPreview(file) {

    console.log(
        "SHOWING PREVIEW"
    );


    const allowedTypes = [
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp"
    ];


    if (
        !allowedTypes.includes(file.type)
    ) {

        alert(
            "Please select PNG, JPG, JPEG or WEBP."
        );

        resetUpload();

        return;
    }


    if (
        file.size >
        10 * 1024 * 1024
    ) {

        alert(
            "Maximum image size is 10 MB."
        );

        resetUpload();

        return;
    }


    const reader =
        new FileReader();


    reader.onload =
        function (event) {

            console.log(
                "IMAGE PREVIEW READY"
            );


            previewImage.src =
                event.target.result;


            fileName.textContent =
                file.name;


            fileSize.textContent =
                formatFileSize(
                    file.size
                );


            uploadPlaceholder.style.display =
                "none";


            previewContainer.style.display =
                "block";


            analyzeBtn.disabled =
                false;


            analyzeBtn.style.opacity =
                "1";


            analyzeBtn.style.cursor =
                "pointer";


            console.log(
                "ANALYZE BUTTON ENABLED"
            );


            analyzeWafer();

        };


    reader.onerror =
        function () {

            console.error(
                "FILE READER ERROR"
            );

            alert(
                "Unable to read image."
            );

        };


    reader.readAsDataURL(file);

}


// ============================================================
// FILE SIZE
// ============================================================

function formatFileSize(bytes) {

    if (bytes < 1024) {

        return bytes + " B";

    }


    if (
        bytes <
        1024 * 1024
    ) {

        return (
            (bytes / 1024).toFixed(1) +
            " KB"
        );

    }


    return (
        (bytes / 1024 / 1024).toFixed(2) +
        " MB"
    );

}


// ============================================================
// REMOVE
// ============================================================

removeBtn.addEventListener(
    "click",
    function (event) {

        event.preventDefault();

        event.stopPropagation();

        resetUpload();

    }
);


// ============================================================
// RESET
// ============================================================

function resetUpload() {

    console.log(
        "RESET UPLOAD"
    );


    selectedFile =
        null;


    fileInput.value =
        "";


    previewImage.src =
        "";


    uploadPlaceholder.style.display =
        "block";


    previewContainer.style.display =
        "none";


    analyzeBtn.disabled =
        true;


    analyzeBtn.style.opacity =
        "0.6";

}


// ============================================================
// DRAG OVER
// ============================================================

uploadBox.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        uploadBox.classList.add(
            "dragging"
        );

    }
);


// ============================================================
// DRAG LEAVE
// ============================================================

uploadBox.addEventListener(
    "dragleave",
    function () {

        uploadBox.classList.remove(
            "dragging"
        );

    }
);


// ============================================================
// DROP
// ============================================================

uploadBox.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        event.stopPropagation();


        uploadBox.classList.remove(
            "dragging"
        );


        const file =
            event.dataTransfer.files[0];


        if (!file) {

            return;

        }


        console.log(
            "DROPPED FILE:",
            file.name
        );


        selectedFile =
            file;


        showPreview(file);

    }
);


// ============================================================
// ANALYZE BUTTON
// ============================================================

analyzeBtn.addEventListener(
    "click",
    async function (event) {

        event.preventDefault();

        event.stopPropagation();


        console.log("");
        console.log(
            "🔥🔥🔥 ANALYZE BUTTON CLICKED 🔥🔥🔥"
        );
        console.log("");


        if (!selectedFile) {

            alert(
                "Please upload a wafer image first."
            );

            return;
        }


        await analyzeWafer();

    }
);


// ============================================================
// ANALYZE WAFER
// ============================================================

async function analyzeWafer() {

    console.log(
        "STARTING WAFER ANALYSIS..."
    );


    // ========================================================
    // SHOW LOADING
    // ========================================================

    loadingSection.style.display =
        "block";


    resultsSection.style.display =
        "none";


    analyzeBtn.disabled =
        true;


    analyzeBtn.innerHTML =
        "<span>Analyzing...</span><span>⏳</span>";


    // ========================================================
    // FORM DATA
    // ========================================================

    const formData =
        new FormData();


    formData.append(
        "file",
        selectedFile
    );


    try {

        console.log(
            "Sending request:",
            API_URL + "/predict"
        );


        const response =
            await fetch(
                API_URL + "/predict",
                {
                    method: "POST",
                    body: formData
                }
            );


        console.log(
            "HTTP STATUS:",
            response.status
        );


        if (!response.ok) {

            const errorText =
                await response.text();


            throw new Error(
                "Backend Error " +
                response.status +
                ": " +
                errorText
            );

        }


        const data =
            await response.json();


        console.log(
            "===================================="
        );


        console.log(
            "🔥 BACKEND RESPONSE"
        );


        console.log(
            data
        );


        console.log(
            "===================================="
        );


        displayResults(data);

    }


    catch (error) {

        console.error(
            "ANALYSIS ERROR:",
            error
        );


        alert(
            "Analysis failed:\n\n" +
            error.message
        );

    }


    finally {

        loadingSection.style.display =
            "none";


        analyzeBtn.disabled =
            false;


        analyzeBtn.innerHTML =
            "<span>Analyze Wafer</span><span>→</span>";

    }

}


// ============================================================
// DISPLAY RESULTS
// ============================================================

function displayResults(data) {

    console.log(
        "🔥 DISPLAYING RESULTS"
    );


    // ========================================================
    // PREDICTION
    // ========================================================

    prediction.textContent =
        data.prediction ||
        "Unknown";


    // ========================================================
    // CONFIDENCE
    // ========================================================

    const conf =
        Number(
            data.confidence || 0
        );


    confidence.textContent =
        conf.toFixed(2) +
        "%";


    confidenceBar.style.width =
        Math.min(
            conf,
            100
        ) +
        "%";


    // ========================================================
    // SEVERITY
    // ========================================================

    severity.textContent =
        calculateSeverity(
            conf
        );


    // ========================================================
    // ORIGINAL IMAGE
    // ========================================================

    if (
        data.filename
    ) {

        originalImage.src =
            API_URL +
            "/uploads/" +
            encodeURIComponent(
                data.filename
            );

    }


    // ========================================================
    // GRAD-CAM
    // ========================================================

    if (
        data.gradcam_url
    ) {

        gradcamImage.src =
            API_URL +
            data.gradcam_url;

    }


    // ========================================================
    // PROBABILITIES
    // ========================================================

    renderProbabilities(
        data.probabilities ||
        {}
    );


    // ========================================================
    // GEMINI
    // ========================================================

    if (
        data.llm_analysis &&
        data.llm_analysis.trim()
    ) {

        llmAnalysis.textContent =
            data.llm_analysis;

    }

    else {

        llmAnalysis.textContent =
            "Gemini analysis unavailable.";

    }


    // ========================================================
    // SHOW RESULTS
    // ========================================================

    resultsSection.style.display =
        "block";


    resultsSection.style.visibility =
        "visible";


    resultsSection.style.opacity =
        "1";


    console.log(
        "🔥 RESULTS SECTION VISIBLE"
    );


    // ========================================================
    // SCROLL
    // ========================================================

    setTimeout(
        function () {

            resultsSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        },
        300
    );

}


// ============================================================
// SEVERITY
// ============================================================

function calculateSeverity(
    confidenceValue
) {

    if (
        confidenceValue >= 90
    ) {

        return "HIGH";

    }


    if (
        confidenceValue >= 70
    ) {

        return "MEDIUM";

    }


    return "LOW";

}


// ============================================================
// PROBABILITIES
// ============================================================

function renderProbabilities(
    data
) {

    probabilities.innerHTML =
        "";


    const entries =
        Object.entries(data);


    entries.sort(
        function (a, b) {

            return (
                Number(b[1]) -
                Number(a[1])
            );

        }
    );


    entries.forEach(
        function ([label, value]) {

            const percentage =
                Number(value) *
                100;


            const row =
                document.createElement(
                    "div"
                );


            row.className =
                "probability-row";


            row.innerHTML = `
                <div class="probability-info">

                    <span>
                        ${escapeHtml(label)}
                    </span>

                    <span>
                        ${percentage.toFixed(2)}%
                    </span>

                </div>

                <div class="probability-track">

                    <div
                        class="probability-fill"
                        style="width:${Math.min(
                            percentage,
                            100
                        )}%"
                    ></div>

                </div>
            `;


            probabilities.appendChild(
                row
            );

        }
    );

}


// ============================================================
// ESCAPE HTML
// ============================================================

function escapeHtml(value) {

    return String(value)

        .replaceAll(
            "&",
            "&amp;"
        )

        .replaceAll(
            "<",
            "&lt;"
        )

        .replaceAll(
            ">",
            "&gt;"
        )

        .replaceAll(
            '"',
            "&quot;"
        )

        .replaceAll(
            "'",
            "&#039;"
        );

}


// ============================================================
// NEW ANALYSIS
// ============================================================

if (newAnalysisBtn) {

    newAnalysisBtn.addEventListener(
        "click",
        function (event) {

            event.preventDefault();

            event.stopPropagation();


            resultsSection.style.display =
                "none";


            resetUpload();


            window.scrollTo({
                top: 0,
                behavior: "smooth"
            });

        }
    );

}


// ============================================================
// IMAGE ERROR
// ============================================================

if (originalImage) {

    originalImage.addEventListener(
        "error",
        function () {

            console.warn(
                "Original image could not load."
            );

        }
    );

}


if (gradcamImage) {

    gradcamImage.addEventListener(
        "error",
        function () {

            console.warn(
                "Grad-CAM image could not load."
            );

        }
    );

}


// ============================================================
// READY
// ============================================================

loadingSection.style.display =
    "none";


resultsSection.style.display =
    "none";


analyzeBtn.disabled =
    true;


console.log(
    "WAFER GPT READY"
);