import os
import sys

# =========================================================================
# FORCE AMBIL PATH UTAMA (Biar gak error No module named 'src' lagi!)
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
# =========================================================================

import cv2
import numpy as np
import tensorflow as tf
import json
from collections import deque

# --- FIX TOTAL: Menggunakan CVZone dengan sensitivitas lebih responsif (0.5) ---
from cvzone.HandTrackingModule import HandDetector
detector = HandDetector(detectionCon=0.5, maxHands=2)
# =========================================================================

# Ambil modul project dari folder src 
from src.config import MODEL_DIR, SEQUENCE_LENGTH
from src.preprocessing import normalize_landmarks  # <--- WAJIB DIIMPORT UNTUK FIX PREDIKSI
from src.post_processing import PredictionSmoother

def load_labels(filename):
    with open(os.path.join(MODEL_DIR, filename), 'r') as f:
        return {int(k): v for k, v in json.load(f).items()}

# Fungsi helper yang sudah diperbaiki logika Kiri/Kanan akibat efek mirror webcam
def cvzone_to_mediapipe_format(hands_data):
    combined = np.zeros((42, 3))
    if not hands_data:
        return combined

    for hand in hands_data:
        # Kunci Perbaikan Mirroring:
        # Jika kamera mendeteksi kiri ("Left"), aslinya di dunia nyata adalah tangan kanan
        is_right = hand["type"] == "Left"
        lm_list = np.array(hand["lmList"]) # Shape (21, 3)
        
        if is_right:
            combined[21:42] = lm_list  # Tangan Kanan di indeks 21-42
        else:
            combined[0:21] = lm_list   # Tangan Kiri di indeks 0-21
            
    return combined

def main():
    # Load Models & Labels
    print("Memuat model dan label...")
    try:
        static_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'static_model.h5'))
        dynamic_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'dynamic_model.h5'))
        labels_static = load_labels('label_static.json')
        labels_dynamic = load_labels('label_dynamic.json')
    except Exception as e:
        print(f"Error memuat model/label: {e}. Pastikan Anda sudah menjalankan train_static.py dan train_dynamic.py!")
        return

    # Inisialisasi State
    mode = "STATIC" # STATIC atau DYNAMIC
    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    smoother = PredictionSmoother()
    current_prediction = "Menunggu..."

    cap = cv2.VideoCapture(0)
    print("\n=== KONTROL WEBCAM ===")
    print("Tekan '1' untuk Mode Huruf/Angka (Statis)")
    print("Tekan '2' untuk Mode Kata (Dinamis)")
    print("Tekan 'q' untuk Keluar")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1) # Mirror
        
        # CVZone mendeteksi tangan sekaligus menggambar landmark di layar
        hands, frame = detector.findHands(frame, draw=True)

        if hands:
            # 1. Ekstrak koordinat kedua tangan menjadi shape (42, 3)
            combined_landmarks = cvzone_to_mediapipe_format(hands)
            
            # 2. SEBUH TOTAL: Normalisasi data landmark sebelum dilempar ke AI
            normalized_landmarks = normalize_landmarks(combined_landmarks)

            if mode == "STATIC":
                # Gunakan data yang sudah dinormalisasi
                input_data = np.expand_dims(normalized_landmarks, axis=0) # shape (1, 42, 3)
                res = static_model.predict(input_data, verbose=0)[0]
                idx = np.argmax(res)
                current_prediction = smoother.process(labels_static[idx], res[idx])

            elif mode == "DYNAMIC":
                # Untuk mode dinamis, masukkan data ternormalisasi ke dalam buffer sekuens
                sequence_buffer.append(normalized_landmarks)
                
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    input_data = np.expand_dims(np.array(sequence_buffer), axis=0) # shape (1, 30, 42, 3)
                    res = dynamic_model.predict(input_data, verbose=0)[0]
                    idx = np.argmax(res)
                    current_prediction = smoother.process(labels_dynamic[idx], res[idx])
        else:
            if mode == "DYNAMIC":
                sequence_buffer.clear()
            current_prediction = "Tidak ada tangan"
            smoother.clear_buffer()

        # UI: Tampilkan hasil di layar
        cv2.putText(frame, f"MODE: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Prediksi: {current_prediction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Gesture Recognition', frame)

        # Kontrol Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            mode = "STATIC"
            smoother.clear_buffer()
            sequence_buffer.clear()
        elif key == ord('2'):
            mode = "DYNAMIC"
            smoother.clear_buffer()
            sequence_buffer.clear()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()