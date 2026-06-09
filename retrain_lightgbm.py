#!/usr/bin/env python3
"""
Script untuk retrain LightGBM model dengan sklearn versi saat ini.

Ini akan:
1. Load dataset
2. Load Keras feature extractor model yang sudah ada
3. Extract features dari training data
4. Train LightGBM baru
5. Save model dengan sklearn versi baru

Run with: python3.11 retrain_lightgbm.py
"""

import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.linalg
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
from gensim.models import Word2Vec
from tensorflow.keras import Model

# Patch scipy.linalg.triu for gensim compatibility
if not hasattr(scipy.linalg, 'triu'):
    scipy.lialg.triu = np.triu

print("="*60)
print("RETRAIN LIGHTGBM MODEL")
print("="*60)

# ====================================
# 1) LOAD DATASET
# ====================================
print("\n[1] Loading dataset...")
try:
    data = pd.read_excel('dataset lengkap.xlsx')
    print(f"✅ Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
except FileNotFoundError:
    print("❌ Error: 'dataset lengkap.xlsx' not found!")
    exit(1)

# ====================================
# 2) PREPROCESS DATA
# ====================================
print("\n[2] Preprocessing data...")

data = data.fillna(0)
df = data.drop(columns=['No'])

def hapus_satuan_dan_bersihkan(val):
    if isinstance(val, str):
        val = re.sub(r'[^\d.,-]', '', val)
        val = re.sub(r'(?<!^)-', '', val)
        val = val.replace(',', '.')
        try:
            return float(val)
        except ValueError:
            return np.nan
    return val

nutrisi_cols = ['Kemasan', 'Energi', 'Lemak', 'Karbohidrat', 'Gula',
                'Protein', 'Garam', 'Natrium Benzoat']

for col in nutrisi_cols:
    df[col] = df[col].apply(hapus_satuan_dan_bersihkan)

df = df.fillna(0)

# Normalize
numeric_cols = ["Kemasan", "Energi", "Lemak", "Karbohidrat",
                "Gula", "Protein", "Garam", "Natrium Benzoat"]

scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("✅ Data preprocessed and normalized")

# ====================================
# 3) PREPARE INPUTS (Text & Numeric)
# ====================================
print("\n[3] Preparing inputs...")

# Encode labels
label_col = "Resiko"
mapping_manual = {
    "aman": 0,
    "sedang": 1,
    "tinggi": 2
}

print(f"   Unique labels in data: {df[label_col].unique()}")

# Normalize labels to lowercase
df[label_col] = df[label_col].str.lower()
df[label_col] = df[label_col].map(mapping_manual)

# Remove rows with NaN labels (unmapped categories)
df = df.dropna(subset=[label_col])
print(f"   After removing NaN: {len(df)} rows")

y_all = df[label_col].values.astype(int)

# Text data
text_data = df['Komposisi'].values
Xw_all = text_data

# Numeric data
Xn_all = df[numeric_cols].values

print(f"✅ Data prepared: {len(y_all)} samples")

# ====================================
# 4) LOAD KERAS MODELS
# ====================================
print("\n[4] Loading Keras models...")

model_path = "models/"
try:
    base_cnn_bilstm = tf.keras.models.load_model(os.path.join(model_path, "cb1_bab3.keras"))
    print("✅ Keras model loaded")
    
    # Create feature extractor
    try:
        output_layer = base_cnn_bilstm.get_layer("fusion_feat").output
    except ValueError:
        print("   Note: Layer 'fusion_feat' not found, using penultimate layer")
        output_layer = base_cnn_bilstm.layers[-2].output
    
    feat_model = Model(
        inputs=base_cnn_bilstm.inputs,
        outputs=output_layer,
        name="feature_extractor"
    )
    print("✅ Feature extractor model created")
    
    # Load Word2Vec
    w2v_model = Word2Vec.load(os.path.join(model_path, "model_w2v_komposisi.model"))
    print(f"✅ Word2Vec model loaded (vector_size={w2v_model.vector_size})")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)

# ====================================
# 5) TOKENIZE AND CREATE TEXT VECTORS
# ====================================
print("\n[5] Processing text data...")

stopwords_id = {
    'dan','yang','dengan','atau','pada','di','ke','dari','untuk','dalam','sebagai','oleh',
    'tanpa','agar','karena','juga','serta','ini','itu','adalah','lebih','dapat','mengandung',
    'menggunakan','mengolah','bahan','produk','perisa','aroma'
}

def tokenize_and_clean_text(text):
    if pd.isna(text):
        return []
    s = str(text).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    
    hasil = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        t = re.sub(r'[^a-z0-9]', '', t)
        if not t or len(t) < 3 or t.isdigit() or t in stopwords_id:
            continue
        hasil.append(t)
    return hasil

def create_document_vector(tokens, w2v_model, target_dim=50):
    wv = w2v_model.wv
    valid_vectors = [wv[t] for t in tokens if t in wv.key_to_index]
    
    if not valid_vectors:
        return np.zeros(target_dim, dtype=np.float32)
    
    mean_vector = np.mean(valid_vectors, axis=0).astype(np.float32)
    
    if len(mean_vector) > target_dim:
        mean_vector = mean_vector[:target_dim]
    elif len(mean_vector) < target_dim:
        mean_vector = np.pad(mean_vector, (0, target_dim - len(mean_vector)))
    
    return mean_vector

# Process all text
print("   Processing text vectors...")
Xw_vecs = []
for i, text in enumerate(Xw_all):
    tokens = tokenize_and_clean_text(text)
    vec = create_document_vector(tokens, w2v_model, target_dim=50)
    Xw_vecs.append(vec)
    if (i + 1) % 50 == 0:
        print(f"      {i+1}/{len(Xw_all)} texts processed")

Xw_vecs = np.array(Xw_vecs, dtype=np.float32)
print(f"✅ Text vectors created: shape={Xw_vecs.shape}")

# ====================================
# 6) SPLIT DATA
# ====================================
print("\n[6] Splitting data (80/20)...")

Xw_train, Xw_test, Xn_train, Xn_test, y_train, y_test = train_test_split(
    Xw_vecs, Xn_all, y_all,
    test_size=0.2,
    stratify=y_all,
    random_state=42
)

print(f"✅ Train size: {len(y_train)}, Test size: {len(y_test)}")

# ====================================
# 7) EXTRACT FEATURES USING KERAS
# ====================================
print("\n[7] Extracting features using Keras model...")

# Prepare text sequences
Xw_train_seq = Xw_train.reshape(Xw_train.shape[0], 50, 1)
Xw_test_seq = Xw_test.reshape(Xw_test.shape[0], 50, 1)

print("   Extracting training features...")
F_train = feat_model.predict([Xw_train_seq, Xn_train], verbose=0)
print(f"   ✅ F_train shape: {F_train.shape}")

print("   Extracting test features...")
F_test = feat_model.predict([Xw_test_seq, Xn_test], verbose=0)
print(f"   ✅ F_test shape: {F_test.shape}")

# ====================================
# 8) TRAIN LIGHTGBM
# ====================================
print("\n[8] Training LightGBM model...")

n_classes = len(np.unique(y_train))

lgbm_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=n_classes,
    num_leaves=100,
    max_depth=10,
    learning_rate=0.1,
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("   Fitting model...")
lgbm_model.fit(F_train, y_train)
print("✅ Model training completed")

# ====================================
# 9) EVALUATE MODEL
# ====================================
print("\n[9] Evaluating model...")

y_pred_train = lgbm_model.predict(F_train)
y_pred_test = lgbm_model.predict(F_test)

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test, average='weighted')

print(f"\n📊 RESULTS:")
print(f"   Train Accuracy: {accuracy_train:.4f}")
print(f"   Test Accuracy:  {accuracy_test:.4f}")
print(f"   Test F1-Score:  {f1_test:.4f}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred_test, target_names=['aman', 'sedang', 'tinggi']))

# ====================================
# 10) SAVE MODEL
# ====================================
print("\n[10] Saving model...")

backup_path = os.path.join(model_path, "model_lgbm_woa_bab3.joblib.backup")
model_file = os.path.join(model_path, "model_lgbm_woa_bab3.joblib")

# Backup old model if exists
if os.path.exists(model_file):
    import shutil
    shutil.copy(model_file, backup_path)
    print(f"   ✅ Old model backed up to: {backup_path}")

# Save new model
joblib.dump(lgbm_model, model_file)
print(f"   ✅ New model saved to: {model_file}")

# Also save scaler
scaler_path = os.path.join(model_path, "scaler.joblib")
joblib.dump(scaler, scaler_path)
print(f"   ✅ Scaler saved to: {scaler_path}")

print("\n" + "="*60)
print("✅ RETRAIN COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\nNotes:")
print(f"   - Model saved with current sklearn version")
print(f"   - Old model backed up: {backup_path}")
print(f"   - You can now test with debug_prediction.py")
