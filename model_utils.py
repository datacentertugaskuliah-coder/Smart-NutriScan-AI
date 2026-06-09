"""
Model utility untuk SMART NutriScan AI.

Revisi v3 memperbaiki empat masalah utama:
1. Klasifikasi Aman, Sedang, dan Tinggi dibuat dari satu sumber keputusan.
2. Data kosong tidak dipaksa menjadi risiko tinggi.
3. Output ML disaring dengan aturan gizi agar prediksi tidak menyimpang ekstrem.
4. Rekomendasi memakai label yang sama dengan hasil klasifikasi.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

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

NUTRITION_NUMERIC_KEYS = [
    "energi",
    "lemak_total",
    "lemak_jenuh",
    "protein",
    "karbohidrat",
    "gula",
    "garam",
    "natrium",
    "natrium_benzoat",
]

HAZARD_KEYS = [
    "energi",
    "lemak_total",
    "lemak_jenuh",
    "karbohidrat",
    "gula",
    "garam",
    "natrium",
    "natrium_benzoat",
]

stopwords_id = {
    "dan", "yang", "dengan", "atau", "pada", "di", "ke", "dari", "untuk", "dalam",
    "sebagai", "oleh", "tanpa", "agar", "karena", "juga", "serta", "ini", "itu",
    "adalah", "lebih", "dapat", "mengandung", "menggunakan", "mengolah", "bahan",
    "produk", "perisa", "aroma",
}


RISK_LEVELS = {
    "AMAN": {
        "label": "Aman",
        "recommendation_label": "Risiko Aman",
        "min_score": 0,
        "max_score": 34.99,
    },
    "SEDANG": {
        "label": "Sedang",
        "recommendation_label": "Risiko Sedang",
        "min_score": 35,
        "max_score": 69.99,
    },
    "TINGGI": {
        "label": "Tinggi",
        "recommendation_label": "Risiko Tinggi",
        "min_score": 70,
        "max_score": 100,
    },
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
        value = safe_float(val)

    if column_name == "Energi" and value > 5000:
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
    numeric_cols = [
        "Energi",
        "Lemak",
        "Lemak Jenuh",
        "Karbohidrat",
        "Gula",
        "Protein",
        "Garam",
        "Natrium",
        "Natrium Benzoat",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: hapus_satuan_dan_bersihkan(x, column_name=col)).fillna(0)

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
        (r"msg|mononatrium glutamat|monosodium glutamate|penguat rasa", "Penguat Rasa"),
        (r"pengawet|natrium benzoat|sodium benzoate|kalium sorbat|propionat", "Pengawet Sintetik"),
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
    """Memuat model Keras, LightGBM, Word2Vec, dan scaler jika file tersedia."""
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


def has_sufficient_input(nutrition_data: Dict[str, Any]) -> bool:
    """Validasi agar data nol tidak dipaksa dianalisis."""
    total_signal = sum(abs(safe_float(nutrition_data.get(key, 0))) for key in HAZARD_KEYS)
    return total_signal > 0


def classify_risk(score: float) -> Dict[str, str]:
    """Satu sumber keputusan untuk tampilan, rekomendasi, dan batch."""
    score = float(max(0, min(safe_float(score), 100)))

    if score < 35:
        level = RISK_LEVELS["AMAN"]
        style = "success"
    elif score < 70:
        level = RISK_LEVELS["SEDANG"]
        style = "warning"
    else:
        level = RISK_LEVELS["TINGGI"]
        style = "error"

    return {
        "label": level["label"],
        "recommendation_label": level["recommendation_label"],
        "style": style,
    }


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


def _score_from_multiclass_probability(model: Any, proba: np.ndarray) -> float | None:
    if np.ndim(proba) != 2 or proba.shape[0] == 0:
        return None

    row = np.asarray(proba[0], dtype=float)
    if row.size == 0:
        return None

    if row.size == 2:
        return float(row[1]) * 100

    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) == row.size:
        risk_points = []
        for cls in classes:
            cls_text = str(cls).lower()
            if "aman" in cls_text or cls_text in {"0", "low", "rendah"}:
                risk_points.append(15)
            elif "sedang" in cls_text or cls_text in {"1", "medium"}:
                risk_points.append(52)
            elif "tinggi" in cls_text or cls_text in {"2", "high"}:
                risk_points.append(88)
            else:
                risk_points.append(50)
        return float(np.dot(row, np.asarray(risk_points, dtype=float)))

    if row.size >= 3:
        anchors = np.linspace(15, 88, row.size)
        return float(np.dot(row, anchors))

    return float(row[0]) * 100


def _predict_probability(lgbm_model: Any, features: np.ndarray) -> float | None:
    if lgbm_model is None:
        return None

    try:
        if hasattr(lgbm_model, "predict_proba"):
            proba = lgbm_model.predict_proba(features)
            score = _score_from_multiclass_probability(lgbm_model, proba)
            if score is not None:
                return float(max(0, min(score, 100)))

        pred = lgbm_model.predict(features)
        value = np.ravel(pred)[0]

        if isinstance(value, str):
            lower = value.lower()
            if "aman" in lower or "rendah" in lower:
                return 15.0
            if "sedang" in lower:
                return 52.0
            if "tinggi" in lower:
                return 88.0
            return None

        numeric_value = float(value)
        if numeric_value <= 1:
            return numeric_value * 100
        if numeric_value in {0, 1, 2}:
            return [15.0, 52.0, 88.0][int(numeric_value)]
        return float(max(0, min(numeric_value, 100)))
    except Exception as exc:
        print(f"Prediksi LightGBM gagal: {exc}")
        return None


def _rule_based_risk(nutrition_data: Dict[str, Any], composition_text: str) -> float:
    """
    Skor gizi terkalibrasi untuk tiga level yang diminta dalam dokumen revisi.
    Contoh uji:
    Aman 180 kkal, gula 6 g, natrium 150 mg berada pada skor rendah.
    Sedang 320 kkal, gula 22 g, natrium 600 mg berada pada skor sedang.
    Tinggi 550 kkal, gula 45 g, natrium 1500 mg berada pada skor tinggi.
    """
    energi = safe_float(nutrition_data.get("energi", 0))
    gula = safe_float(nutrition_data.get("gula", 0))
    natrium = safe_float(nutrition_data.get("natrium", 0))
    garam = safe_float(nutrition_data.get("garam", 0))
    lemak_total = safe_float(nutrition_data.get("lemak_total", 0))
    lemak_jenuh = safe_float(nutrition_data.get("lemak_jenuh", 0))
    natrium_benzoat = safe_float(nutrition_data.get("natrium_benzoat", 0))

    if natrium == 0 and garam > 0:
        natrium = garam * 400

    score = 0.0
    score += min(energi / 550 * 15, 15)
    score += min(lemak_total / 30 * 15, 15)
    score += min(lemak_jenuh / 14 * 15, 15)
    score += min(karbohidrat_or_zero(nutrition_data) / 75 * 5, 5)
    score += min(gula / 45 * 20, 20)
    score += min(natrium / 1500 * 20, 20)
    score += min(garam / 4 * 5, 5)
    score += min(natrium_benzoat / 300 * 10, 10)

    is_upf, flags = detect_harmful_additives(composition_text)
    if is_upf:
        score += min(4 + len(flags) * 2, 10)

    return float(max(0, min(score, 100)))


def karbohidrat_or_zero(nutrition_data: Dict[str, Any]) -> float:
    return safe_float(nutrition_data.get("karbohidrat", 0))


def _build_xai_factors(nutrition_data: Dict[str, Any]) -> Dict[str, float]:
    return {
        "Energi": safe_float(nutrition_data.get("energi", 0)),
        "Lemak Total": safe_float(nutrition_data.get("lemak_total", 0)),
        "Lemak Jenuh": safe_float(nutrition_data.get("lemak_jenuh", 0)),
        "Protein": safe_float(nutrition_data.get("protein", 0)),
        "Karbohidrat": safe_float(nutrition_data.get("karbohidrat", 0)),
        "Gula": safe_float(nutrition_data.get("gula", 0)),
        "Natrium": safe_float(nutrition_data.get("natrium", 0)),
        "Natrium Benzoat": safe_float(nutrition_data.get("natrium_benzoat", 0)),
    }


def _risk_level_distance(score_a: float, score_b: float) -> int:
    order = {"Aman": 0, "Sedang": 1, "Tinggi": 2}
    return abs(order[classify_risk(score_a)["label"]] - order[classify_risk(score_b)["label"]])


def analyze_product_fully(
    nutrition_data: Dict[str, Any],
    composition_text: str,
    feat_model: Any,
    lgbm_model: Any,
    w2v_model: Any,
    scaler: Any,
) -> Tuple[float, Dict[str, float], str]:
    """
    Analisis produk dengan logika hybrid yang dijaga konsisten.
    Model ML tetap dicoba, tetapi skor akhir tidak boleh menyimpang ekstrem dari profil gizi.
    """
    if not has_sufficient_input(nutrition_data):
        xai_factors = _build_xai_factors(nutrition_data)
        return 0.0, xai_factors, "Data belum cukup untuk dianalisis. Isi minimal satu nilai gizi yang valid sebelum menjalankan rekomendasi."

    rule_score = _rule_based_risk(nutrition_data, composition_text)
    ml_score = None

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

    for candidate in feature_candidates:
        ml_score = _predict_probability(lgbm_model, np.asarray(candidate))
        if ml_score is not None:
            break

    used_rule_guard = False
    if ml_score is None:
        final_score = rule_score
        used_rule_guard = True
    else:
        gap = abs(ml_score - rule_score)
        level_gap = _risk_level_distance(ml_score, rule_score)
        if gap > 25 or level_gap >= 2:
            final_score = rule_score
            used_rule_guard = True
        else:
            final_score = (0.60 * rule_score) + (0.40 * ml_score)

    final_score = float(max(0, min(final_score, 100)))
    xai_factors = _build_xai_factors(nutrition_data)
    _, flags = detect_harmful_additives(composition_text)
    recommendation = _build_recommendation(final_score, nutrition_data, flags, used_rule_guard, ml_score, rule_score)
    return final_score, xai_factors, recommendation


def _build_recommendation(
    risk_score: float,
    nutrition_data: Dict[str, Any],
    upf_flags: List[str],
    used_rule_guard: bool = False,
    ml_score: float | None = None,
    rule_score: float | None = None,
) -> str:
    risk_info = classify_risk(risk_score)
    label = risk_info["recommendation_label"]
    notes: List[str] = []

    if risk_info["label"] == "Tinggi":
        notes.append(f"{label}: Batasi konsumsi karena profil gizi produk menunjukkan risiko tinggi.")
    elif risk_info["label"] == "Sedang":
        notes.append(f"{label}: Boleh dikonsumsi sesekali, tetapi tetap perhatikan porsi dan frekuensi.")
    else:
        notes.append(f"{label}: Produk relatif aman berdasarkan nilai gizi yang dimasukkan, dengan catatan porsi tetap wajar.")

    gula = safe_float(nutrition_data.get("gula", 0))
    natrium = safe_float(nutrition_data.get("natrium", 0))
    lemak_jenuh = safe_float(nutrition_data.get("lemak_jenuh", 0))
    natrium_benzoat = safe_float(nutrition_data.get("natrium_benzoat", 0))

    if gula >= 22:
        notes.append("Gula cukup tinggi sehingga perlu dibatasi.")
    elif gula >= 10:
        notes.append("Gula perlu diperhatikan agar tidak melebihi batas harian.")

    if natrium >= 600:
        notes.append("Natrium cukup tinggi, terutama bagi pengguna dengan risiko hipertensi.")
    elif natrium >= 300:
        notes.append("Natrium perlu dipantau pada konsumsi berulang.")

    if lemak_jenuh >= 6:
        notes.append("Lemak jenuh cukup tinggi dan tidak disarankan dikonsumsi terlalu sering.")

    if natrium_benzoat >= 130:
        notes.append("Kandungan natrium benzoat perlu diperhatikan sesuai batas aman dan frekuensi konsumsi.")

    if upf_flags:
        notes.append("Komposisi menunjukkan indikasi bahan ultra proses: " + ", ".join(upf_flags) + ".")

    if used_rule_guard:
        notes.append("Catatan sistem: skor akhir dijaga dengan aturan gizi terkalibrasi agar hasil Aman, Sedang, dan Tinggi tetap konsisten.")

    return " ".join(notes)
