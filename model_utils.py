"""
Model utility untuk Smart NutriScan AI.
File ini menjaga nama fungsi lama agar app.py tetap kompatibel, sekaligus menambah fallback agar aplikasi tidak mati jika artefak model belum cocok di cloud.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import scipy.linalg

if not hasattr(scipy.linalg, "triu"):
    scipy.linalg.triu = np.triu

try:
    import tensorflow as tf
    from tensorflow.keras import Model
except Exception:
    tf = None
    Model = None

try:
    from gensim.models import Word2Vec
except Exception:
    Word2Vec = None

try:
    from sklearn.preprocessing import MinMaxScaler
except Exception:
    MinMaxScaler = None


NUMERIC_ORDER = [
    "Kemasan",
    "Energi",
    "Lemak",
    "Karbohidrat",
    "Gula",
    "Protein",
    "Garam",
    "Natrium Benzoat",
]

APP_TO_MODEL_COLUMN = {
    "kemasan": "Kemasan",
    "energi": "Energi",
    "lemak_total": "Lemak",
    "karbohidrat": "Karbohidrat",
    "gula": "Gula",
    "protein": "Protein",
    "garam": "Garam",
    "natrium_benzoat": "Natrium Benzoat",
}


stopwords_id = {
    "dan", "yang", "dengan", "atau", "pada", "di", "ke", "dari", "untuk", "dalam",
    "sebagai", "oleh", "tanpa", "agar", "karena", "juga", "serta", "ini", "itu",
    "adalah", "lebih", "dapat", "mengandung", "menggunakan", "mengolah", "bahan",
    "produk", "perisa", "aroma",
}


def hapus_satuan_dan_bersihkan(val: Any, column_name: str | None = None) -> float:
    """Membersihkan angka nutrisi dari satuan seperti g, mg, kkal, dan kJ."""
    if pd.isna(val):
        return 0.0

    if isinstance(val, str):
        raw = val.strip().lower()
        is_less_than = "<" in raw
        raw = re.sub(r"[^\d.,-]", "", raw)

        if raw.count(",") > 0 and raw.count(".") > 0:
            if raw.rfind(",") > raw.rfind("."):
                raw = raw.replace(".", "").replace(",", ".")
            else:
                raw = raw.replace(",", "")
        else:
            raw = raw.replace(",", ".")

        if raw in {"", ".", "-"}:
            return 0.0

        try:
            value = float(raw)
        except ValueError:
            return 0.0

        if is_less_than:
            value = value / 2.0
    else:
        try:
            value = float(val)
        except (TypeError, ValueError):
            return 0.0

    if column_name == "Energi" and value > 500:
        value = value / 4.184

    return float(value)


def get_scaler():
    """Membuat scaler dari dataset jika scaler joblib tidak tersedia."""
    if MinMaxScaler is None:
        return None

    try:
        data = pd.read_excel("dataset lengkap.xlsx").fillna(0)
        df = data.drop(columns=["No"], errors="ignore")

        for col in NUMERIC_ORDER:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: hapus_satuan_dan_bersihkan(x, column_name=col))
            else:
                df[col] = 0.0

        scaler = MinMaxScaler()
        scaler.fit(df[NUMERIC_ORDER])
        return scaler
    except Exception as exc:
        print(f"Scaler fallback gagal dibuat: {exc}")
        return None


def preprocess_batch_excel_data(df: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan kolom numerik pada file Excel batch."""
    df = df.copy()
    numeric_cols = ["Energi", "Lemak", "Karbohidrat", "Gula", "Protein", "Garam", "Natrium Benzoat"]
    existing_cols = [col for col in numeric_cols if col in df.columns]

    for col in existing_cols:
        df[col] = df[col].apply(lambda x: hapus_satuan_dan_bersihkan(x, column_name=col))

    if existing_cols:
        df[existing_cols] = df[existing_cols].fillna(0)

    return df


def filtering_tokens(tokens: List[str], min_len: int = 3, remove_numbers: bool = True) -> List[str]:
    hasil: List[str] = []

    for token in tokens:
        token = token.strip().lower()
        token = re.sub(r"[^a-z0-9]", "", token)

        if not token:
            continue
        if remove_numbers and token.isdigit():
            continue
        if len(token) < min_len:
            continue
        if token in stopwords_id:
            continue

        hasil.append(token)

    return hasil


def tokenize_and_clean_text(text: str) -> List[str]:
    if pd.isna(text):
        return []

    value = str(text).lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return filtering_tokens(value.split())


def detect_harmful_additives(text: str) -> Tuple[bool, List[str]]:
    """Deteksi sederhana bahan ultra proses dari teks komposisi."""
    if pd.isna(text) or str(text).strip() == "":
        return False, []

    lower_text = str(text).lower()
    red_flags: List[str] = []

    checks = [
        (r"aspartam|sukralosa|sakarin|asesulfam|siklamat|pemanis buatan", "Pemanis Buatan"),
        (r"tartrazin|merah allura|kuning fcf|biru berlian|pewarna sintetik", "Pewarna Sintetik"),
        (r"msg|mononatrium glutamat|penguat rasa", "Penguat Rasa"),
        (r"pengawet|natrium benzoat|kalium sorbat|propionat", "Pengawet Sintetik"),
        (r"sirup fruktosa|fructose syrup|corn syrup|hfcs", "Sirup Fruktosa Tinggi"),
        (r"minyak nabati terhidrogenasi|lemak trans|hydrogenated", "Lemak Trans"),
    ]

    for pattern, label in checks:
        if re.search(pattern, lower_text) and label not in red_flags:
            red_flags.append(label)

    return len(red_flags) > 0, red_flags


def create_document_vector(tokens: List[str], w2v_model: Any, target_dim: int = 50) -> np.ndarray:
    """Membuat vektor dokumen dari Word2Vec."""
    if w2v_model is None or not hasattr(w2v_model, "wv"):
        return np.zeros(target_dim, dtype=np.float32)

    try:
        word_vectors = w2v_model.wv
        valid_vectors = [word_vectors[token] for token in tokens if token in word_vectors.key_to_index]

        if not valid_vectors:
            return np.zeros(target_dim, dtype=np.float32)

        mean_vector = np.mean(valid_vectors, axis=0).astype(np.float32)
        if len(mean_vector) > target_dim:
            mean_vector = mean_vector[:target_dim]
        elif len(mean_vector) < target_dim:
            mean_vector = np.pad(mean_vector, (0, target_dim - len(mean_vector)))

        return mean_vector.astype(np.float32)
    except Exception:
        return np.zeros(target_dim, dtype=np.float32)


def load_prediction_models():
    """Memuat model Keras, LightGBM, Word2Vec, dan scaler."""
    model_path = "models"
    feat_model = None
    lgbm_model = None
    w2v_model = None
    scaler = None

    try:
        if tf is not None:
            keras_path = os.path.join(model_path, "cb1_bab3.keras")
            if os.path.exists(keras_path):
                base_model = tf.keras.models.load_model(keras_path)
                try:
                    output_layer = base_model.get_layer("fusion_feat").output
                except Exception:
                    output_layer = base_model.layers[-2].output if len(base_model.layers) >= 2 else base_model.output
                feat_model = Model(inputs=base_model.inputs, outputs=output_layer, name="feature_extractor")
    except Exception as exc:
        print(f"Model Keras gagal dimuat: {exc}")

    try:
        lgbm_path = os.path.join(model_path, "model_lgbm_woa_bab3.joblib")
        if os.path.exists(lgbm_path):
            lgbm_model = joblib.load(lgbm_path)
    except Exception as exc:
        print(f"Model LightGBM gagal dimuat: {exc}")

    try:
        w2v_path = os.path.join(model_path, "model_w2v_komposisi.model")
        if Word2Vec is not None and os.path.exists(w2v_path):
            w2v_model = Word2Vec.load(w2v_path)
    except Exception as exc:
        print(f"Model Word2Vec gagal dimuat: {exc}")

    try:
        scaler_path = os.path.join(model_path, "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = get_scaler()
    except Exception as exc:
        print(f"Scaler gagal dimuat: {exc}")
        scaler = get_scaler()

    return feat_model, lgbm_model, w2v_model, scaler


def _nutrition_to_model_array(nutrition_data: Dict[str, Any]) -> np.ndarray:
    values = {
        "Kemasan": nutrition_data.get("kemasan", 0),
        "Energi": nutrition_data.get("energi", 0),
        "Lemak": nutrition_data.get("lemak_total", 0),
        "Karbohidrat": nutrition_data.get("karbohidrat", 0),
        "Gula": nutrition_data.get("gula", 0),
        "Protein": nutrition_data.get("protein", 0),
        "Garam": nutrition_data.get("garam", 0),
        "Natrium Benzoat": nutrition_data.get("natrium_benzoat", 0),
    }

    numeric = [hapus_satuan_dan_bersihkan(values[col], column_name=col) for col in NUMERIC_ORDER]
    return np.array([numeric], dtype=np.float32)


def _scale_numeric(numeric_array: np.ndarray, scaler: Any) -> np.ndarray:
    if scaler is None:
        return numeric_array.astype(np.float32)

    try:
        return scaler.transform(numeric_array).astype(np.float32)
    except Exception:
        return numeric_array.astype(np.float32)


def _predict_probability(lgbm_model: Any, features: np.ndarray) -> float | None:
    if lgbm_model is None:
        return None

    try:
        if hasattr(lgbm_model, "predict_proba"):
            proba = lgbm_model.predict_proba(features)
            if np.ndim(proba) == 2 and proba.shape[1] > 1:
                return float(proba[0, 1]) * 100
            return float(np.ravel(proba)[0]) * 100

        pred = lgbm_model.predict(features)
        value = float(np.ravel(pred)[0])
        return value * 100 if value <= 1 else value
    except Exception as exc:
        print(f"Prediksi LightGBM gagal: {exc}")
        return None


def _rule_based_risk(nutrition_data: Dict[str, Any], composition_text: str) -> float:
    energi = float(nutrition_data.get("energi", 0) or 0)
    gula = float(nutrition_data.get("gula", 0) or 0)
    natrium = float(nutrition_data.get("natrium", 0) or 0)
    lemak_total = float(nutrition_data.get("lemak_total", 0) or 0)
    lemak_jenuh = float(nutrition_data.get("lemak_jenuh", 0) or 0)
    natrium_benzoat = float(nutrition_data.get("natrium_benzoat", 0) or 0)
    is_upf, flags = detect_harmful_additives(composition_text)

    score = 0.0
    score += min(energi / 400 * 20, 20)
    score += min(gula / 25 * 25, 25)
    score += min(natrium / 600 * 20, 20)
    score += min(lemak_total / 20 * 15, 15)
    score += min(lemak_jenuh / 10 * 15, 15)
    score += min(natrium_benzoat / 100 * 10, 10)

    if is_upf:
        score += min(5 + len(flags) * 3, 15)

    return float(max(0, min(score, 100)))


def analyze_product_fully(
    nutrition_data: Dict[str, Any],
    composition_text: str,
    feat_model: Any,
    lgbm_model: Any,
    w2v_model: Any,
    scaler: Any,
) -> Tuple[float, Dict[str, float], str]:
    """Analisis produk dengan model hybrid jika tersedia, lalu fallback rule based jika model tidak siap."""
    numeric = _nutrition_to_model_array(nutrition_data)
    numeric_scaled = _scale_numeric(numeric, scaler)

    tokens = tokenize_and_clean_text(composition_text)
    text_vec = create_document_vector(tokens, w2v_model, target_dim=50).reshape(1, -1)

    feature_candidates: List[np.ndarray] = []

    if feat_model is not None:
        try:
            feature_candidates.append(feat_model.predict([numeric_scaled, text_vec], verbose=0))
        except Exception:
            pass
        try:
            joined_input = np.concatenate([numeric_scaled, text_vec], axis=1)
            feature_candidates.append(feat_model.predict(joined_input, verbose=0))
        except Exception:
            pass

    feature_candidates.append(np.concatenate([numeric_scaled, text_vec], axis=1))
    feature_candidates.append(numeric_scaled)

    risk_score = None
    for candidate in feature_candidates:
        risk_score = _predict_probability(lgbm_model, np.asarray(candidate))
        if risk_score is not None:
            break

    used_fallback = False
    if risk_score is None:
        risk_score = _rule_based_risk(nutrition_data, composition_text)
        used_fallback = True

    risk_score = float(max(0, min(risk_score, 100)))

    xai_factors = {
        "Energi": float(nutrition_data.get("energi", 0) or 0),
        "Lemak Total": float(nutrition_data.get("lemak_total", 0) or 0),
        "Lemak Jenuh": float(nutrition_data.get("lemak_jenuh", 0) or 0),
        "Protein": float(nutrition_data.get("protein", 0) or 0),
        "Karbohidrat": float(nutrition_data.get("karbohidrat", 0) or 0),
        "Gula": float(nutrition_data.get("gula", 0) or 0),
        "Natrium": float(nutrition_data.get("natrium", 0) or 0),
        "Natrium Benzoat": float(nutrition_data.get("natrium_benzoat", 0) or 0),
    }

    is_upf, flags = detect_harmful_additives(composition_text)
    recommendation = _build_recommendation(risk_score, nutrition_data, flags, used_fallback)
    return risk_score, xai_factors, recommendation


def _build_recommendation(
    risk_score: float,
    nutrition_data: Dict[str, Any],
    upf_flags: List[str],
    used_fallback: bool = False,
) -> str:
    notes: List[str] = []

    if risk_score >= 75:
        notes.append("Batasi konsumsi. Produk ini masuk kategori risiko sangat tinggi berdasarkan profil gizi yang terbaca.")
    elif risk_score >= 50:
        notes.append("Konsumsi sebaiknya dibatasi. Produk ini menunjukkan risiko gizi cukup tinggi.")
    elif risk_score >= 25:
        notes.append("Konsumsi masih mungkin dilakukan, tetapi tetap perlu kontrol porsi.")
    else:
        notes.append("Produk relatif lebih aman secara gizi, dengan catatan porsi tetap dikendalikan.")

    gula = float(nutrition_data.get("gula", 0) or 0)
    natrium = float(nutrition_data.get("natrium", 0) or 0)
    lemak_jenuh = float(nutrition_data.get("lemak_jenuh", 0) or 0)

    if gula >= 10:
        notes.append("Kandungan gula perlu diperhatikan, terutama untuk pengguna yang menjaga asupan gula harian.")
    if natrium >= 300:
        notes.append("Kandungan natrium cukup menonjol, sehingga perlu dibatasi pada pengguna dengan risiko hipertensi.")
    if lemak_jenuh >= 5:
        notes.append("Lemak jenuh cukup tinggi dan tidak disarankan dikonsumsi terlalu sering.")
    if upf_flags:
        notes.append("Komposisi menunjukkan indikasi bahan ultra proses: " + ", ".join(upf_flags) + ".")
    if used_fallback:
        notes.append("Catatan sistem: model prediksi utama belum berhasil digunakan, sehingga aplikasi memakai analisis cadangan berbasis aturan gizi.")

    return " ".join(notes)
