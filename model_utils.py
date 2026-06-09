import os
import re
import joblib  # use standalone joblib package
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.linalg

# Patch scipy.linalg.triu for gensim compatibility
if not hasattr(scipy.linalg, 'triu'):
    scipy.linalg.triu = np.triu

from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model
from tensorflow.keras.initializers import Orthogonal

# ==========================================
# 1. DATA ENGINEERING & CLEANING MODULE
# ==========================================

def hapus_satuan_dan_bersihkan(val, column_name=None):
    """
    Cleans numerical values by removing units, handling both comma and dot decimal separators.
    [UPGRADE EXPERT]: Menangani kasus edge-case seperti "< 1g" atau "~5mg".
    
    Handles cases like:
    - "100g" -> 100.0
    - "< 1g" -> 0.5 (Heuristik Data Science untuk batas bawah)
    - "<5mg" -> 2.5
    - "1.234,56" -> 1234.56
    """
    if isinstance(val, str):
        val = val.strip().lower()

        # Penanganan khusus tanda "kurang dari" (<)
        # Jika "< 1g", secara konservatif kita anggap setengahnya agar tidak 0 tapi juga tidak 1
        is_less_than = False
        if '<' in val:
            is_less_than = True

        # Remove non-numeric characters except dots, commas, and minus sign
        val = re.sub(r'[^\d.,-]', '', val)
        val = re.sub(r'(?<!^)-', '', val) # Remove minus signs except at the beginning
        
        if ',' in val and '.' in val:
            if val.rindex(',') > val.rindex('.'):
                val = val.replace('.', '').replace(',', '.')
            else:
                val = val.replace(',', '.')
        else:
            val = val.replace(',', '.')
        
        try:
            if val == '': return 0.0
            result = float(val)
            
            # Apply heuristic for less than
            if is_less_than:
                result = result / 2.0

            if column_name == 'Energi' and result > 500:
                result = result / 4.184
                # print(f"Note: Converted energy value {val} Kj to {result:.1f} kkal")
            
            return result
        except ValueError:
            return np.nan
    
    try:
        result = float(val)
        if column_name == 'Energi' and result > 500:
            result = result / 4.184
        return result
    except (ValueError, TypeError):
        return np.nan

def get_scaler():
    """
    Loads the original dataset to fit and return the MinMaxScaler.
    This is crucial for ensuring the input data is scaled exactly
    as the training data was.
    """
    try:
        data = pd.read_excel('dataset lengkap.xlsx')
        data = data.fillna(0)
        df = data.drop(columns=['No'])

        nutrisi_cols = ['Kemasan', 'Energi', 'Lemak', 'Karbohidrat', 'Gula',
                        'Protein', 'Garam', 'Natrium Benzoat']

        for col in nutrisi_cols:
            df[col] = df[col].apply(lambda x: hapus_satuan_dan_bersihkan(x, column_name=col))
        
        df = df.fillna(0)

        numeric_cols = [
            "Kemasan", "Energi", "Lemak", "Karbohidrat",
            "Gula", "Protein", "Garam", "Natrium Benzoat"
        ]
        scaler = MinMaxScaler()
        scaler.fit(df[numeric_cols])
        return scaler
    except Exception as e:
        print(f"Error creating scaler: {e}")
        return None

def preprocess_batch_excel_data(df):
    """
    Preprocesses batch Excel data by cleaning all numerical columns.
    Handles units (g, mg, kkal, Kj, etc.), comma decimals, and mixed formats.
    
    Special handling:
    - Energi: Converts Kj to kkal if value > 500 (detected as Kj)
    - All columns: Removes units, handles comma/dot decimal separators
    
    Args:
        df (pd.DataFrame): DataFrame read from Excel with nutrition columns
        
    Returns:
        pd.DataFrame: DataFrame with cleaned numerical values
    """
    df = df.copy()
    
    # Nutrition columns that need cleaning (only process if they exist)
    numeric_cols = ['Energi', 'Lemak', 'Karbohidrat', 'Gula', 'Protein', 'Garam', 'Natrium Benzoat']
    existing_cols = [col for col in numeric_cols if col in df.columns]
    
    for col in existing_cols:
        # Pass column name for special handling (e.g., Kj to kkal conversion)
        df[col] = df[col].apply(lambda x: hapus_satuan_dan_bersihkan(x, column_name=col))
    
    # Fill any NaN values with 0 (only in existing columns)
    df[existing_cols] = df[existing_cols].fillna(0)
    
    return df

# ==========================================
# 2. NLP & TEXT MINING MODULE
# ==========================================

stopwords_id = {
    'dan','yang','dengan','atau','pada','di','ke','dari','untuk','dalam','sebagai','oleh',
    'tanpa','agar','karena','juga','serta','ini','itu','adalah','lebih','dapat','mengandung',
    'menggunakan','mengolah','bahan','produk','perisa','aroma'
}

def filtering_tokens(tokens, min_len=3, remove_numbers=True):
    hasil = []
    for t in tokens:
        t = t.strip()
        if not t: continue
        t = re.sub(r'[^a-z0-9]', '', t)
        if not t: continue
        if remove_numbers and t.isdigit(): continue
        if len(t) < min_len: continue
        if t in stopwords_id: continue
        hasil.append(t)
    return hasil

def tokenize_and_clean_text(text: str):
    if pd.isna(text): return []
    s = str(text).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return filtering_tokens(s.split())

def detect_harmful_additives(text: str):
    """
    [FITUR EXPERT]: Natural Language Processing (Rule-based)
    Mendeteksi bahan-bahan Ultra-Processed Food (UPF) dari teks komposisi.
    Ini menambah value "Explainability" pada sistem AI.
    """
    if pd.isna(text) or text == "":
        return False, []

    text = str(text).lower()
    red_flags = []

    # Kamus NLP sederhana untuk deteksi zat aditif berisiko
    if re.search(r'aspartam|sukralosa|sakarin|asesulfam|siklamat|pemanis buatan', text):
        red_flags.append("Pemanis Buatan")
    if re.search(r'tartrazin|merah allura|kuning fcf|biru berlian|pewarna sintetik', text):
        red_flags.append("Pewarna Sintetik")
    if re.search(r'msg|mononatrium glutamat|penguat rasa', text):
        red_flags.append("Penguat Rasa (MSG)")
    if re.search(r'pengawet|natrium benzoat|kalium sorbat|propionat', text):
        red_flags.append("Pengawet Sintetik")
    if re.search(r'sirup fruktosa|fructose syrup|corn syrup|hfcs', text):
        red_flags.append("High-Fructose Corn Syrup (Risiko Obesitas)")
    if re.search(r'minyak nabati terhidrogenasi|lemak trans|hydrogenated', text):
        red_flags.append("Lemak Trans / Minyak Terhidrogenasi")

    is_upf = len(red_flags) > 0
    return is_upf, red_flags

def create_document_vector(tokens, w2v_model, target_dim=50):
    """Creates a document vector by averaging word vectors."""
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


# ==========================================
# 3. MACHINE LEARNING & PREDICTION MODULE
# ==========================================

def load_prediction_models():
    """Loads all models and the fitted scaler."""
    model_path = "models/"
    try:
        base_cnn_bilstm = tf.keras.models.load_model(os.path.join(model_path, "cb1_bab3.keras"))
        
        try:
            output_layer = base_cnn_bilstm.get_layer("fusion_feat").output
        except ValueError:
            print(f"Layer 'fusion_feat' not found. Using layer: {base_cnn_bilstm.layers[-2].name}")
            output_layer = base_cnn_bilstm.layers[-2].output

        feat_model = Model(
            inputs=base_cnn_bilstm.inputs,
            outputs=output_layer,
            name="feature_extractor"
        )

        lgbm_model = joblib.load(os.path.join(model_path, "model_lgbm_woa_bab3.joblib"))
        w2v_model = Word2Vec.load(os.path.join(model_path, "model_w2v_komposisi.model"))
        
        scaler_path = os.path.join(model_path, "scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            print("Warning: scaler.joblib not found. Using dummy scaler.")
            scaler = MinMaxScaler()
            scaler.fit(np.zeros((1, 8)))

        print("✅ All models and scaler loaded successfully.")
        return feat_model, lgbm_model, w2v_model, scaler

    except Exception as e:
        print(f"An error occurred while loading models: {e}")
        return None, None, None, None

def predict_with_lgbm(model, features):
    """Wrapper untuk LightGBM prediction."""
    return model.predict_proba(features)

def analyze_product_fully(nutrition_data, composition_text, feat_model, lgbm_model, w2v_model, scaler):
    """
    Analyzes a product by performing the full hybrid pipeline (Keras + LGBM).
    [UPGRADE EXPERT]: Menambahkan integrasi Text Mining untuk UPF detection
    dan rekomendasi yang jauh lebih pintar.
    """
    try:
        # --- 1. PREPARE NUMERICAL INPUT ---
        numeric_cols_order = ["Kemasan", "Energi", "Lemak", "Karbohidrat", "Gula", "Protein", "Garam", "Natrium Benzoat"]
        
        input_numeric_df = pd.DataFrame([{
            "Kemasan": 0,
            "Energi": nutrition_data.get('energi', 0),
            "Lemak": nutrition_data.get('lemak_total', 0),
            "Karbohidrat": nutrition_data.get('karbohidrat', 0),
            "Gula": nutrition_data.get('gula', 0),
            "Protein": nutrition_data.get('protein', 0),
            "Garam": nutrition_data.get('garam', 0),
            "Natrium Benzoat": nutrition_data.get('natrium_benzoat', 0)
        }], columns=numeric_cols_order)

        scaled_numeric_input = scaler.transform(input_numeric_df).astype(np.float32)

        # --- 2. PREPARE TEXT INPUT & UPF DETECTION ---
        tokens = tokenize_and_clean_text(composition_text)
        doc_vector = create_document_vector(tokens, w2v_model, target_dim=50)
        text_input_seq = doc_vector.reshape(1, 50, 1)

        # NLP Rule-Based Scan for Additives
        is_upf, found_additives = detect_harmful_additives(composition_text)

        # --- 3. FEATURE EXTRACTION ---
        extracted_features = feat_model.predict([text_input_seq, scaled_numeric_input], verbose=0)

        # --- 4. FINAL PREDICTION ---
        prediction_proba = predict_with_lgbm(lgbm_model, extracted_features)
        
        # Risk Score (0-100)
        risk_score = (prediction_proba[0][1] * 50) + (prediction_proba[0][2] * 100)

        # --- 5. GENERATE EXPLANATIONS (XAI) AND RECOMMENDATIONS ---
        # Untuk Hybrid CNN, raw XAI susah ditarik langsung, kita gunakan porsi komposisi
        # relatif terhadap standar harian (Health Informatics rules) untuk proxy
        xai_factors = {
            'Gula (g)': nutrition_data.get('gula', 0),
            'Natrium (mg)': nutrition_data.get('natrium', 0) or nutrition_data.get('garam', 0) * 1000,
            'Lemak Total (g)': nutrition_data.get('lemak_total', 0),
            'Energi Total (kkal)': nutrition_data.get('energi', 0),
            'Natrium Benzoat (mg)': nutrition_data.get('natrium_benzoat', 0)
        }
        sorted_factors = dict(sorted(xai_factors.items(), key=lambda item: item[1], reverse=True))

        # Smart Recommendation Generation based on ML Probabilities AND Text Mining Flags
        pred_class = np.argmax(prediction_proba[0])

        # Base Recommendation from ML
        if pred_class == 2: # 'tinggi'
            recommendation = "🛑 **Risiko TINGGI (ML Prediction):** Sangat tidak disarankan untuk konsumsi harian. "
        elif pred_class == 1: # 'sedang'
            recommendation = "⚠️ **Risiko SEDANG (ML Prediction):** Boleh dikonsumsi sesekali, perhatikan porsi sajian Anda. "
        else: # 'aman'
            recommendation = "✅ **Risiko RENDAH (ML Prediction):** Relatif aman sebagai bagian dari diet seimbang. "

        # Injecting NLP Domain Knowledge into Recommendation
        if is_upf:
            recommendation += f"\n\n🔬 **Insight Teks Komposisi:** Sistem mendeteksi produk ini termasuk **Ultra-Processed Food (UPF)** karena mengandung aditif sintetik: *{', '.join(found_additives)}*. "
            if pred_class == 0:
                recommendation += "Meskipun kalorinya mungkin aman, paparan aditif kimia jangka panjang perlu diwaspadai."
            else:
                recommendation += "Kombinasi makronutrien buruk dan aditif kimia membuatnya sangat tidak sehat."
        else:
            if composition_text.strip() != "" and composition_text.lower() != "tidak terdeteksi.":
                recommendation += "\n\n🍃 **Insight Teks Komposisi:** Tidak terdeteksi aditif kimia ekstrem. Komposisinya relatif alami/bersih."
            
        return risk_score, sorted_factors, recommendation

    except Exception as e:
        print(f"Error during full analysis: {e}")
        return 50.0, {}, f"Gagal melakukan analisis penuh: {e}"


def analyze_product_fully_debug(nutrition_data, composition_text, feat_model, lgbm_model, w2v_model, scaler):
    """Debug version of analyze_product_fully"""
    try:
        numeric_cols_order = ["Kemasan", "Energi", "Lemak", "Karbohidrat", "Gula", "Protein", "Garam", "Natrium Benzoat"]
        input_numeric_df = pd.DataFrame([{
            "Kemasan": 0, "Energi": nutrition_data.get('energi', 0), "Lemak": nutrition_data.get('lemak_total', 0),
            "Karbohidrat": nutrition_data.get('karbohidrat', 0), "Gula": nutrition_data.get('gula', 0),
            "Protein": nutrition_data.get('protein', 0), "Garam": nutrition_data.get('garam', 0),
            "Natrium Benzoat": nutrition_data.get('natrium_benzoat', 0)
        }], columns=numeric_cols_order)

        print("\n=== DEBUG: Numerical Input ===")
        print(input_numeric_df)
        
        scaled_numeric_input = scaler.transform(input_numeric_df).astype(np.float32)
        print("\n=== DEBUG: Scaled Numerical Input ===")
        print(scaled_numeric_input)
        
        tokens = tokenize_and_clean_text(composition_text)
        print(f"\n=== DEBUG: Tokens ({len(tokens)}) ===")
        print(tokens[:20])
        
        doc_vector = create_document_vector(tokens, w2v_model, target_dim=50)
        text_input_seq = doc_vector.reshape(1, 50, 1)

        extracted_features = feat_model.predict([text_input_seq, scaled_numeric_input], verbose=0)
        print(f"\n=== DEBUG: Extracted Features ===")
        print(f"First 10 features: {extracted_features[0][:10]}")
        
        prediction_proba = predict_with_lgbm(lgbm_model, extracted_features)
        print(f"\n=== DEBUG: LightGBM Probabilities ===")
        print(f"P(aman={0}): {prediction_proba[0][0]:.4f} | P(sedang={1}): {prediction_proba[0][1]:.4f} | P(tinggi={2}): {prediction_proba[0][2]:.4f}")
        
        risk_score = (prediction_proba[0][1] * 50) + (prediction_proba[0][2] * 100)
        print(f"Risk Score = {risk_score:.2f}%")
        
        return risk_score, prediction_proba[0]

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None
