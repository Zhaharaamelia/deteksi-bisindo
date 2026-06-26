// VisiSign - MediaPipe Hands Detection + FastAPI Integration
// FIX: Tambah Camera object, debounce cooldown, async predict, auto-reset label

document.addEventListener('DOMContentLoaded', async () => {

    console.log('MediaPipe Hands Module Loaded');

    const API_HOSTS = [
        'http://127.0.0.1:8001',
        'http://127.0.0.1:8000'
    ];
    let apiHost = API_HOSTS[0];
    let apiOnline = false;

    const getApiUrl = () => `${apiHost}/predict`;

    const checkApiHost = async () => {
        for (const host of API_HOSTS) {
            try {
                const response = await fetch(`${host}/health`, { method: 'GET' });
                if (response.ok) {
                    apiHost = host;
                    apiOnline = true;
                    console.log(`✅ API host set to ${host}`);
                    return;
                }
            } catch (error) {
                // abaikan dan coba host berikutnya
            }
        }
        apiOnline = false;
        console.warn('⚠️ API tidak dapat dijangkau pada semua host');
    };

    await checkApiHost();

    const webcamElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('canvas');

    const predictionText   = document.getElementById('prediction-text');
    const confidenceText   = document.getElementById('confidence-text');
    const btnClearHistory  = document.getElementById('btn-clear-history');
    const btnResetPrediction = document.getElementById('btn-reset-prediction');
    const btnToggleMode    = document.getElementById('btn-toggle-mode');
    const historyLog       = document.getElementById('history-log');

    if (!webcamElement || !canvasElement) {
        console.error('❌ Webcam atau Canvas tidak ditemukan di DOM');
        return;
    }

    const canvasCtx = canvasElement.getContext('2d');

    // =====================================================
    // STATE
    // =====================================================

    let sequence       = [];
    const SEQUENCE_LENGTH = 30;
    let currentMode    = "DYNAMIC";

    const PREDICT_COOLDOWN_MS  = 200;
    let lastPredictTime        = 0;

    const PREDICTION_RESET_MS  = 2500;
    let predictionResetTimer   = null;

    let lastLoggedWord  = "";
    let lastLoggedTime  = 0;

    // =====================================================
    // MEDIAPIPE HANDS
    // =====================================================

    const hands = new Hands({
        locateFile: (file) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.5
    });

    // =====================================================
    // ✅ FIX: Camera object — inilah yang mengirim setiap
    // frame dari webcam ke hands.send() agar landmark muncul.
    // Tanpa ini, hands.onResults() tidak pernah dipanggil.
    // =====================================================

    const camera = new Camera(webcamElement, {
        onFrame: async () => {
            await hands.send({ image: webcamElement });
        },
        width: 1280,
        height: 720
    });

    camera.start();
    console.log('✅ MediaPipe Camera started');

    // =====================================================
    // NORMALIZATION
    // =====================================================

    const normalizeHandLandmarks = (landmarks) => {
        if (!landmarks || landmarks.length === 0) {
            return new Array(63).fill(0);
        }
        const wrist  = landmarks[0];
        const output = [];
        landmarks.forEach((lm) => {
            output.push(
                lm.x - wrist.x,
                lm.y - wrist.y,
                lm.z - wrist.z
            );
        });
        return output;
    };

    // =====================================================
    // EXTRACT KEYPOINTS
    // =====================================================

    const extractHandKeypoints = (results) => {
        let leftHand  = new Array(63).fill(0);
        let rightHand = new Array(63).fill(0);

        if (results.multiHandLandmarks && results.multiHandedness) {
            results.multiHandLandmarks.forEach((landmarks, index) => {
                const handLabel  = results.multiHandedness[index].label;
                const normalized = normalizeHandLandmarks(landmarks);
                if (handLabel === "Left") {
                    leftHand  = normalized;
                } else {
                    rightHand = normalized;
                }
            });
        }

        return [...leftHand, ...rightHand];
    };

    // =====================================================
    // API REQUEST
    // =====================================================

    const sendPredictionRequest = async (seqData) => {
        const now = Date.now();
        if (now - lastPredictTime < PREDICT_COOLDOWN_MS) return;
        if (!seqData || seqData.length === 0) return;

        if (!apiOnline) {
            await checkApiHost();
        }

        if (!apiOnline) {
            if (predictionText) predictionText.textContent = 'PREDIKSI: ERROR API';
            if (confidenceText) {
                confidenceText.className =
                    'text-sm font-semibold text-red-500 flex items-center justify-center gap-2';
                confidenceText.innerHTML =
                    `<i class="fa-solid fa-server"></i>
                    FastAPI tidak berjalan (cek port ${API_HOSTS.join(' atau ')})`;
            }
            return;
        }

        lastPredictTime = now;

        try {
            const response = await fetch(getApiUrl(), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sequence: seqData, mode: currentMode })
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error("API ERROR:", response.status, errorText);
                if (predictionText) predictionText.textContent = "PREDIKSI: ERROR";
                if (confidenceText) {
                    confidenceText.className =
                        'text-sm font-semibold text-red-500 flex items-center justify-center gap-2';
                    confidenceText.innerHTML =
                        `<i class="fa-solid fa-circle-exclamation"></i> HTTP ${response.status}`;
                }
                return;
            }

            const data = await response.json();
            updateUI(data.prediction, data.confidence);

        } catch (error) {
            console.error("Gagal terhubung ke FastAPI:", error);
            if (predictionText) predictionText.textContent = 'PREDIKSI: ERROR API';
            if (confidenceText) {
                confidenceText.className =
                    'text-sm font-semibold text-red-500 flex items-center justify-center gap-2';
                confidenceText.innerHTML =
                    `<i class="fa-solid fa-server"></i> FastAPI tidak berjalan`;
            }
        }
    };

    // =====================================================
    // UPDATE UI
    // =====================================================

    const resetPredictionDisplay = () => {
        if (predictionText) predictionText.textContent = 'PREDIKSI: -';
        if (confidenceText) {
            confidenceText.className =
                'text-sm font-semibold text-amber-500 flex items-center justify-center gap-2';
            confidenceText.innerHTML =
                `<i class="fa-solid fa-hand"></i> Menunggu gerakan...`;
        }
    };

    const updateUI = (label, confidence) => {
        if (!predictionText || !confidenceText) return;

        const ignoredLabels = ["Menunggu...", "Tidak ada tangan", "Menunggu gerakan..."];

        if (!ignoredLabels.includes(label)) {
            predictionText.textContent = `PREDIKSI: ${label.toUpperCase()}`;
            confidenceText.className =
                'text-sm font-semibold text-emerald-600 flex items-center justify-center gap-2';
            confidenceText.innerHTML =
                `<i class="fa-solid fa-circle-check"></i> Akurasi: ${confidence}`;

            const now = Date.now();
            if (label !== lastLoggedWord || now - lastLoggedTime > 3000) {
                addToHistory(label);
                lastLoggedWord = label;
                lastLoggedTime = now;
            }

            clearTimeout(predictionResetTimer);
            predictionResetTimer = setTimeout(resetPredictionDisplay, PREDICTION_RESET_MS);

        } else {
            predictionText.textContent = 'PREDIKSI: -';
            confidenceText.className =
                'text-sm font-semibold text-amber-500 flex items-center justify-center gap-2';
            confidenceText.innerHTML =
                `<i class="fa-solid fa-hand"></i> ${label}`;
            clearTimeout(predictionResetTimer);
        }
    };

    // =====================================================
    // HISTORY
    // =====================================================

    const addToHistory = (word) => {
        if (!historyLog) return;

        const placeholder = historyLog.querySelector('.italic');
        if (placeholder) placeholder.remove();

        const now  = new Date();
        const time = now.toLocaleTimeString('id-ID');

        const item = document.createElement('div');
        item.className =
            'py-2.5 flex justify-between items-center text-slate-900 border-b border-slate-100';
        item.innerHTML = `
            <span class="flex items-center gap-2 font-bold text-teal-600">
                <i class="fa-solid fa-clock text-slate-300"></i>
                ${word.toUpperCase()}
            </span>
            <span class="text-[10px] font-mono text-slate-400">${time}</span>
        `;

        historyLog.insertBefore(item, historyLog.firstChild);
        while (historyLog.children.length > 10) {
            historyLog.removeChild(historyLog.lastChild);
        }
    };

    // =====================================================
    // HANDS CALLBACK — dipanggil setiap frame oleh Camera
    // =====================================================

    hands.onResults((results) => {

        // Sesuaikan ukuran canvas dengan video
        if (
            canvasElement.width  !== webcamElement.videoWidth ||
            canvasElement.height !== webcamElement.videoHeight
        ) {
            canvasElement.width  = webcamElement.videoWidth;
            canvasElement.height = webcamElement.videoHeight;
        }

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        // Gambar landmark tangan
        if (results.multiHandLandmarks) {
            results.multiHandLandmarks.forEach((landmarks) => {
                drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                    color: '#10B981',
                    lineWidth: 3
                });
                drawLandmarks(canvasCtx, landmarks, {
                    color: '#FA5252',
                    radius: 3
                });
            });
        }

        canvasCtx.restore();

        // Kumpulkan keypoints ke sequence
        const keypoints = extractHandKeypoints(results);
        sequence.push(keypoints);

        const maxBuffer = currentMode === "STATIC" ? 1 : SEQUENCE_LENGTH;
        while (sequence.length > maxBuffer) {
            sequence.shift();
        }

        sendPredictionRequest(sequence);
    });

    // =====================================================
    // MODE BUTTON
    // =====================================================

    if (btnToggleMode) {
        btnToggleMode.addEventListener('click', (e) => {
            e.preventDefault();
            sequence = [];
            clearTimeout(predictionResetTimer);
            resetPredictionDisplay();

            if (currentMode === "DYNAMIC") {
                currentMode = "STATIC";
                btnToggleMode.innerHTML = `
                    <span class="w-7 h-7 bg-pink-600 text-white rounded-lg flex items-center justify-center text-[11px]">
                        <i class="fa-solid fa-font"></i>
                    </span>
                    <div>
                        <p>Mode: Huruf/Angka</p>
                        <p class="text-[9px] text-pink-500 font-normal">Ubah ke kata dinamis</p>
                    </div>
                `;
            } else {
                currentMode = "DYNAMIC";
                btnToggleMode.innerHTML = `
                    <span class="w-7 h-7 bg-green-600 text-white rounded-lg flex items-center justify-center text-[11px]">
                        <i class="fa-solid fa-exchange-alt"></i>
                    </span>
                    <div>
                        <p>Mode: Kata</p>
                        <p class="text-[9px] text-green-500 font-normal">Ubah ke huruf/angka</p>
                    </div>
                `;
            }

            console.log("Mode:", currentMode);
        });
    }

    // =====================================================
    // RESET BUTTON
    // =====================================================

    if (btnResetPrediction) {
        btnResetPrediction.addEventListener('click', (e) => {
            e.preventDefault();
            sequence        = [];
            lastLoggedWord  = "";
            lastPredictTime = 0;
            clearTimeout(predictionResetTimer);
            resetPredictionDisplay();
        });
    }

    // =====================================================
    // CLEAR HISTORY
    // =====================================================

    if (btnClearHistory) {
        btnClearHistory.addEventListener('click', (e) => {
            e.preventDefault();
            historyLog.innerHTML =
                `<div class="py-2.5 text-center text-slate-400 text-[11px] italic">
                    Belum ada kata terdeteksi
                </div>`;
            lastLoggedWord = "";
        });
    }

});