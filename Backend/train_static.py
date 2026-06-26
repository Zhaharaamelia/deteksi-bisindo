import os
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import dari folder src
from src.config import STATIC_DATA_DIR, MODEL_DIR, STATIC_MODEL_PATH
from src.model_architectures import build_static_model

def train():
    print("Memuat data statis...")
    X_path = os.path.join(STATIC_DATA_DIR, 'X_static.npy')
    y_path = os.path.join(STATIC_DATA_DIR, 'y_static.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Error: Data tidak ditemukan di {STATIC_DATA_DIR}. Jalankan extract_features.py dulu!")
        return

    X = np.load(X_path)
    y = np.load(y_path)

    # Encode label teks ('A', 'B') menjadi angka (0, 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Simpan mapping label ke JSON agar bisa dibaca saat inference webcam
    os.makedirs(MODEL_DIR, exist_ok=True)
    label_mapping = {int(index): str(label) for index, label in enumerate(le.classes_)}
    with open(os.path.join(MODEL_DIR, 'label_static.json'), 'w') as f:
        json.dump(label_mapping, f)

    # Split data training dan testing (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print(f"Jumlah kelas: {num_classes}")
    print(f"Data latih: {X_train.shape}, Data uji: {X_test.shape}")

    # Bangun dan latih model
    model = build_static_model(num_classes)
    
    # Callback untuk berhenti jika akurasi sudah bagus
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("\nMulai pelatihan model statis...")
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

    # Simpan model
    model.save(STATIC_MODEL_PATH)
    print(f"\nModel statis berhasil disimpan di: {STATIC_MODEL_PATH}")

if __name__ == "__main__":
    train()