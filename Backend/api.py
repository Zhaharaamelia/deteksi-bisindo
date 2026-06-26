import os
import sys
import json
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =====================================================
# PATH SETUP
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.config import MODEL_DIR, SEQUENCE_LENGTH
from src.post_processing import PredictionSmoother

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="VisiSign API v2",
    version="2.1",
    description="API baru untuk prediksi bahasa isyarat dengan fitur model info dan smoothing."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# THREAD EXECUTOR (agar model.predict tidak blocking)
# =====================================================

executor = ThreadPoolExecutor(max_workers=2)

# =====================================================
# MODEL LOADING
# =====================================================

STATIC_MODEL_PATH = os.path.join(MODEL_DIR, "static_model.h5")
DYNAMIC_MODEL_PATH = os.path.join(MODEL_DIR, "dynamic_model.h5")

static_model = None
dynamic_model = None


def load_labels(filename: str) -> Dict[int, str]:
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


labels_static = {}
labels_dynamic = {}

smoother = PredictionSmoother()


def safe_load_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        print(f"[ERROR] Load model failed: {model_path}")
        print(exc)
        return None


print("\n==============================")
print("Starting VisiSign API v2")
print("==============================")

static_model = safe_load_model(STATIC_MODEL_PATH)
dynamic_model = safe_load_model(DYNAMIC_MODEL_PATH)

try:
    labels_static = load_labels("label_static.json")
    labels_dynamic = load_labels("label_dynamic.json")
except Exception as exc:
    print("[ERROR] Failed loading labels:")
    print(exc)

# =====================================================
# WARMUP — jalankan saat server start agar TF graph
# ter-compile sebelum request pertama user masuk
# =====================================================

@app.on_event("startup")
async def warmup_models():
    loop = asyncio.get_event_loop()

    if static_model is not None:
        dummy_static = np.zeros((1, 42, 3), dtype=np.float32)
        await loop.run_in_executor(
            executor,
            lambda: static_model.predict(dummy_static, verbose=0)
        )
        print("[INFO] Static model warmed up.")

    if dynamic_model is not None:
        dummy_dynamic = np.zeros((1, SEQUENCE_LENGTH, 42, 3), dtype=np.float32)
        await loop.run_in_executor(
            executor,
            lambda: dynamic_model.predict(dummy_dynamic, verbose=0)
        )
        print("[INFO] Dynamic model warmed up.")

    print("[INFO] Warmup selesai. API siap menerima request.")

# =====================================================
# DATA MODELS
# =====================================================


class PredictionRequest(BaseModel):
    sequence: Any
    mode: str = "STATIC"


class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]


class PredictionResponse(BaseModel):
    prediction: str
    confidence: str
    confidence_value: float
    mode: str


class ModelInfo(BaseModel):
    static_model_loaded: bool
    dynamic_model_loaded: bool
    static_labels: int
    dynamic_labels: int


# =====================================================
# HELPERS
# =====================================================


def normalize_input_sequence(sequence: Any) -> np.ndarray:
    if not isinstance(sequence, list):
        raise HTTPException(status_code=400, detail="sequence harus berupa array")

    if len(sequence) == 0:
        raise HTTPException(status_code=400, detail="sequence kosong")

    try:
        seq_array = np.array(sequence, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"sequence tidak dapat diubah menjadi array: {exc}"
        )

    return seq_array


async def process_prediction(sequence: Any, mode: str) -> PredictionResponse:
    global smoother

    loop = asyncio.get_event_loop()
    mode = mode.upper()
    seq_array = normalize_input_sequence(sequence)

    if mode == "STATIC":
        if static_model is None:
            raise HTTPException(status_code=500, detail="Static model gagal dimuat")

        last_frame = seq_array[-1]
        if len(last_frame.shape) == 1 and last_frame.shape[0] == 126:
            last_frame = last_frame.reshape(42, 3)

        input_data = np.expand_dims(last_frame, axis=0)

        # Jalankan predict di thread terpisah agar tidak blocking event loop
        predictions = await loop.run_in_executor(
            executor,
            lambda: static_model.predict(input_data, verbose=0)[0]
        )

        idx = int(np.argmax(predictions))
        confidence = float(predictions[idx])
        label = labels_static.get(idx, "UNKNOWN")

    elif mode == "DYNAMIC":
        if dynamic_model is None:
            raise HTTPException(status_code=500, detail="Dynamic model gagal dimuat")

        if len(seq_array.shape) == 2 and seq_array.shape[1] == 126:
            if len(seq_array) < SEQUENCE_LENGTH:
                pad_len = SEQUENCE_LENGTH - len(seq_array)
                padding = np.zeros((pad_len, 126), dtype=np.float32)
                seq_array = np.vstack((padding, seq_array))
            else:
                seq_array = seq_array[-SEQUENCE_LENGTH:]
            seq_array = seq_array.reshape(SEQUENCE_LENGTH, 42, 3)

        elif len(seq_array.shape) == 3:
            if len(seq_array) < SEQUENCE_LENGTH:
                pad_len = SEQUENCE_LENGTH - len(seq_array)
                padding = np.zeros((pad_len, 42, 3), dtype=np.float32)
                seq_array = np.concatenate((padding, seq_array), axis=0)
            else:
                seq_array = seq_array[-SEQUENCE_LENGTH:]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Shape tidak dikenali: {seq_array.shape}"
            )

        input_data = np.expand_dims(seq_array, axis=0)

        # Jalankan predict di thread terpisah agar tidak blocking event loop
        predictions = await loop.run_in_executor(
            executor,
            lambda: dynamic_model.predict(input_data, verbose=0)[0]
        )

        idx = int(np.argmax(predictions))
        confidence = float(predictions[idx])
        label = labels_dynamic.get(idx, "UNKNOWN")

    else:
        raise HTTPException(status_code=400, detail="Mode harus STATIC atau DYNAMIC")

    label = smoother.process(label, confidence)

    return PredictionResponse(
        prediction=label,
        confidence=f"{confidence * 100:.2f}%",
        confidence_value=confidence,
        mode=mode,
    )


# =====================================================
# ROUTES
# =====================================================


@app.get("/")
def index():
    return {"status": "online", "service": "VisiSign API v2"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "static_model": static_model is not None,
        "dynamic_model": dynamic_model is not None,
        "label_counts": {
            "static": len(labels_static),
            "dynamic": len(labels_dynamic),
        },
    }


@app.get("/models", response_model=ModelInfo)
def get_models():
    return ModelInfo(
        static_model_loaded=static_model is not None,
        dynamic_model_loaded=dynamic_model is not None,
        static_labels=len(labels_static),
        dynamic_labels=len(labels_dynamic),
    )


@app.get("/labels")
def get_labels(
    mode: Optional[str] = Query("STATIC", description="Pilih mode: STATIC atau DYNAMIC")
):
    mode = mode.upper()
    if mode == "STATIC":
        return {"mode": "STATIC", "labels": labels_static}
    if mode == "DYNAMIC":
        return {"mode": "DYNAMIC", "labels": labels_dynamic}
    raise HTTPException(status_code=400, detail="mode harus STATIC atau DYNAMIC")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    return await process_prediction(request.sequence, request.mode)


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    responses = [
        await process_prediction(item.sequence, item.mode)
        for item in request.requests
    ]
    return {"predictions": [r.dict() for r in responses]}


@app.post("/smoother/clear")
def clear_smoother():
    smoother.clear_buffer()
    return {"status": "ok", "message": "Smoother buffer cleared"}


@app.get("/modes")
def get_modes():
    return {"available_modes": ["STATIC", "DYNAMIC"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
    )