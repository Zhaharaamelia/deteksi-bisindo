import os
import sys
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==========================================
# PERBAIKAN PATH (PENTING)
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Ambil konfigurasi & arsitektur dengan penanganan multi-path agar aman
try:
    from config import DYNAMIC_DATA_DIR, MODEL_DIR, DYNAMIC_MODEL_PATH
    from model_architectures import build_dynamic_model
except ModuleNotFoundError:
    from src.config import DYNAMIC_DATA_DIR, MODEL_DIR, DYNAMIC_MODEL_PATH
    from src.model_architectures import build_dynamic_model

def train():
    print("Memuat data dinamis...")
    X_path = os.path.join(DYNAMIC_DATA_DIR, 'X_dynamic.npy')
    y_path = os.path.join(DYNAMIC_DATA_DIR, 'y_dynamic.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"Error: Data tidak ditemukan di {DYNAMIC_DATA_DIR}. Jalankan extract_features.py dulu!")
        return

    X = np.load(X_path)
    y = np.load(y_path)

    # Encode label teks ('Halo', 'Tolong') menjadi angka (0, 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Simpan mapping label ke JSON
    os.makedirs(MODEL_DIR, exist_ok=True)
    label_mapping = {int(index): str(label) for index, label in enumerate(le.classes_)}
    with open(os.path.join(MODEL_DIR, 'label_dynamic.json'), 'w') as f:
        json.dump(label_mapping, f)

    # Split data training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print(f"Jumlah kelas: {num_classes}")
    print(f"Data latih: {X_train.shape}, Data uji: {X_test.shape}")

    # Bangun dan latih model
    model = build_dynamic_model(num_classes)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    print("\nMulai pelatihan model dinamis...")
    model.fit(X_train, y_train, epochs=150, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stop])

    # Simpan model
    model.save(DYNAMIC_MODEL_PATH)
    print(f"\nModel dinamis berhasil disimpan di: {DYNAMIC_MODEL_PATH}")

if __name__ == "__main__":
    train()