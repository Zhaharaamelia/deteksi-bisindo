import os
import sys

# ==========================================
# 1. SETUP PATH (Mencegah ModuleNotFound)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import base64
import json
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2

# Import modul dari folder src
from src.config import MODEL_DIR, SEQUENCE_LENGTH
from src.preprocessing import extract_two_hands
from src.post_processing import PredictionSmoother

app = FastAPI(title="Sign Language Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. MUAT MODEL & LABEL
# ==========================================
print("Memuat model AI...")
STATIC_MODEL_PATH = os.path.join(MODEL_DIR, 'static_model.h5')
DYNAMIC_MODEL_PATH = os.path.join(MODEL_DIR, 'dynamic_model.h5')

if not os.path.exists(STATIC_MODEL_PATH) or not os.path.exists(DYNAMIC_MODEL_PATH):
    raise FileNotFoundError("File model .h5 tidak ditemukan! Jalankan train_static.py & train_dynamic.py dulu.")

static_model = tf.keras.models.load_model(STATIC_MODEL_PATH, compile=False)
dynamic_model = tf.keras.models.load_model(DYNAMIC_MODEL_PATH, compile=False)

def load_labels(filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, 'r') as f:
        return {int(k): v for k, v in json.load(f).items()}

labels_static = load_labels('label_static.json')
labels_dynamic = load_labels('label_dynamic.json')

# ==========================================
# 3. INISIALISASI MEDIAPIPE
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# ==========================================
# 4. WEBSOCKET ENDPOINT
# ==========================================
@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client terhubung!")
    
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    smoother = PredictionSmoother()
    
    try:
        while True:
            data = await websocket.receive_json()
            image_data = data.get("image", "")
            mode = data.get("mode", "STATIC")
            
            if not image_data: continue
                
            # Decode Base64
            encoded_data = image_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Preprocessing
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = hands.process(frame_rgb)
            current_prediction = "Menunggu..."
            
            if results.multi_hand_landmarks:
                combined_landmarks = extract_two_hands(results.multi_hand_landmarks)
                
                if mode == "STATIC":
                    input_data = np.expand_dims(combined_landmarks, axis=0)
                    res = static_model.predict(input_data, verbose=0)[0]
                    idx = np.argmax(res)
                    current_prediction = smoother.process(labels_static[idx], res[idx])
                    
                elif mode == "DYNAMIC":
                    sequence_buffer.append(combined_landmarks)
                    if len(sequence_buffer) == SEQUENCE_LENGTH:
                        input_data = np.expand_dims(np.array(sequence_buffer), axis=0)
                        res = dynamic_model.predict(input_data, verbose=0)[0]
                        idx = np.argmax(res)
                        current_prediction = smoother.process(labels_dynamic[idx], res[idx])
            else:
                if mode == "DYNAMIC": sequence_buffer.clear()
                smoother.clear_buffer()
                current_prediction = "Tidak ada tangan"
                
            await websocket.send_json({
                "prediction": current_prediction,
                "mode": mode
            })
            
    except WebSocketDisconnect:
        print("Client terputus.")

if __name__ == "__main__":
    import uvicorn
    # Jalankan dengan "app:app" karena file ini bernama app.py
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)