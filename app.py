import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import re
import cv2
import difflib
import plotly.express as px
import plotly.graph_objects as go

import scipy.linalg

# Patch scipy.linalg.triu for gensim compatibility
if not hasattr(scipy.linalg, 'triu'):
    scipy.linalg.triu = np.triu

# Import the new model utility functions
from model_utils import load_prediction_models, analyze_product_fully, preprocess_batch_excel_data

import easyocr

# Initialize session state for history
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="SMART NutriScan AI",
    page_icon="assets/Logo Smart NutriScan AI.png",
    layout="wide"
)

# --- Memuat Model ---
# Use the new loading function from model_utils.py
@st.cache_resource
def load_all_models_and_scaler():
    """Fungsi untuk memuat semua model AI dan scaler."""
    feat_model, lgbm_model, w2v_model, scaler = load_prediction_models()
    if feat_model and lgbm_model and w2v_model and scaler:
        st.success("Model AI dan scaler berhasil dimuat.")
        return feat_model, lgbm_model, w2v_model, scaler
    else:
        st.error("Gagal memuat satu atau lebih komponen AI. Aplikasi mungkin tidak berfungsi dengan benar.")
        return None, None, None, None

@st.cache_resource
def load_ocr_model():
    """Fungsi untuk memuat model OCR."""
    reader = easyocr.Reader(['id', 'en']) # 'id' for Indonesian, 'en' for English
    st.success("Model OCR (EasyOCR) berhasil dimuat.")
    return reader

feat_model, lgbm_model, w2v_model, scaler = load_all_models_and_scaler()
reader = load_ocr_model()



def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil Analisis')
    processed_data = output.getvalue()
    return processed_data


def preprocess_image_for_ocr(pil_image):
    """
    [EXPERT COMPUTER VISION]: Preprocessing gambar tabel nutrisi.
    Menggunakan OpenCV untuk mengatasi blur, glare (kilau), dan meningkatkan kontras teks.
    """
    open_cv_image = np.array(pil_image)
    if len(open_cv_image.shape) > 2 and open_cv_image.shape[2] == 4:
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2RGB)
    if len(open_cv_image.shape) == 3:
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = open_cv_image

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    thresh = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    return Image.fromarray(opening)


def extract_value_near_keyword(words_list, target_keywords, is_text_search=False):
    """
    [EXPERT NLP]: Fuzzy Matching untuk mengekstrak angka dari tabel berantakan.
    """
    best_match = None
    match_idx = -1

    for i, word in enumerate(words_list):
        matches = difflib.get_close_matches(word.lower(), target_keywords, n=1, cutoff=0.7)
        if matches:
            best_match = word
            match_idx = i
            break

    if match_idx == -1:
        return "" if is_text_search else 0.0

    if is_text_search:
        rest_of_text = " ".join(words_list[match_idx+1:])
        return rest_of_text

    search_window = words_list[match_idx+1 : match_idx+8]
    for w in search_window:
        cleaned_num = re.sub(r'[^\d.,]', '', w)
        if cleaned_num.endswith('.') or cleaned_num.endswith(','):
            cleaned_num = cleaned_num[:-1]
        cleaned_num = cleaned_num.replace(',', '.')
        try:
            val = float(cleaned_num)
            return val
        except ValueError:
            continue
    return 0.0


def parse_nutrition_text(detected_text_list, is_composition_only=False):
    """
    [UPGRADED]: Menganalisis list string (output OCR asli) menggunakan Fuzzy Search.
    Bisa mem-parsing seluruh nilai gizi, atau komposisi saja berdasarkan parameter is_composition_only.
    """
    raw_text = " ".join(detected_text_list).replace('\n', ' ')
    words = raw_text.split()
    data = {}

    if not is_composition_only:
        data['energi'] = extract_value_near_keyword(words, ['energi', 'energy', 'kalori', 'calories'])
        data['lemak_total'] = extract_value_near_keyword(words, ['lemak', 'fat'])
        data['lemak_jenuh'] = extract_value_near_keyword(words, ['jenuh', 'saturated'])
        data['protein'] = extract_value_near_keyword(words, ['protein'])
        data['karbohidrat'] = extract_value_near_keyword(words, ['karbohidrat', 'carbohydrate', 'karbo'])
        data['gula'] = extract_value_near_keyword(words, ['gula', 'sugar', 'sukrosa'])
        data['garam'] = extract_value_near_keyword(words, ['garam', 'salt'])
        data['natrium'] = extract_value_near_keyword(words, ['natrium', 'sodium'])
        data['natrium_benzoat'] = extract_value_near_keyword(words, ['benzoat', 'pengawet benzoat'])

        if data.get('garam', 0) > 0 and data.get('natrium', 0) == 0:
            data['natrium'] = data['garam'] * 400

    # Logic Ekstraksi Komposisi (selalu dijalankan untuk case di mana kedua bagian ada di 1 tabel)
    komposisi_raw = extract_value_near_keyword(words, ['komposisi', 'ingredients', 'bahan-bahan', 'bahan'], is_text_search=True)
    if komposisi_raw:
        komposisi_text = komposisi_raw.strip()
        komposisi_text = re.split(r"mengandung alergen|diproduksi menggunakan|informasi nilai gizi", komposisi_text, flags=re.IGNORECASE)[0]
        data['komposisi'] = komposisi_text.strip().capitalize()
    else:
        # Fallback regex, dan jika ini khusus scan komposisi, asumsikan *semua* teks adalah komposisi
        if is_composition_only and len(words) > 2:
            data['komposisi'] = raw_text.strip().capitalize()
        else:
            komposisi_match = re.search(r"(?:komposisi|ingredients|daftar bahan)\s*:\s*(.*?)(?:\.|$)", raw_text.lower())
            if komposisi_match:
                data['komposisi'] = komposisi_match.group(1).strip().capitalize()
            else:
                data['komposisi'] = "Tidak terdeteksi."
        
    if not is_composition_only:
        if len(words) >= 3:
            data['product_name'] = " ".join(words[:3]).title()
        else:
            data['product_name'] = "Produk Tanpa Nama"

    return data

# --- FUNGSI HELPER UNTUK VISUALISASI BI & HEALTH METRICS ---
def render_holistic_nutrition_metrics(energi, takaran_saji, lemak_total, karbohidrat, protein, gula, natrium, lemak_jenuh, current_threshold, user_profile):
    st.markdown("---")
    st.markdown("### 📊 Profil Gizi & Makronutrien Holistik")
    st.write("Analisis mendalam mengenai sumber kalori dan dampak glikemik berdasarkan takaran saji.")

    # 1. Row Atas: Kepadatan Energi & Rasio Glikemik
    metrik_col1, metrik_col2 = st.columns(2)

    with metrik_col1:
        kepadatan_energi = energi / takaran_saji if takaran_saji > 0 else 0

        # Klasifikasi Kepadatan Energi
        if kepadatan_energi > 4:
            kepadatan_status = "🔴 Sangat Tinggi (Padat Kalori)"
            kepadatan_color = "inverse"
        elif kepadatan_energi > 1.5:
            kepadatan_status = "🟡 Tinggi"
            kepadatan_color = "off"
        elif kepadatan_energi > 0.6:
            kepadatan_status = "🟢 Rendah (Ideal)"
            kepadatan_color = "normal"
        else:
            kepadatan_status = "🔵 Sangat Rendah"
            kepadatan_color = "normal"

        st.metric(label="Kepadatan Energi (kkal/gram)", value=f"{kepadatan_energi:.1f}", delta=kepadatan_status, delta_color=kepadatan_color)
        st.caption("Menunjukkan seberapa padat kalori dalam produk ini. Kepadatan tinggi memicu obesitas jika tidak dikontrol.")

    with metrik_col2:
        rasio_gula = (gula / karbohidrat) * 100 if karbohidrat > 0 else 0
        if rasio_gula > 50:
            rasio_status = "🔴 Tinggi Gula Sederhana"
            rasio_color = "inverse"
        elif rasio_gula > 25:
            rasio_status = "🟡 Waspada Glikemik"
            rasio_color = "off"
        else:
            rasio_status = "🟢 Karbohidrat Kompleks"
            rasio_color = "normal"

        st.metric(label="Rasio Gula dari Total Karbohidrat", value=f"{rasio_gula:.1f}%", delta=rasio_status, delta_color=rasio_color)
        st.caption("Jika >50%, sebagian besar karbohidrat adalah gula sederhana yang bisa memicu lonjakan gula darah (*sugar spike*).")

    # 2. Row Tengah: Distribusi Makronutrien (Pie Chart)
    kalori_lemak = lemak_total * 9
    kalori_karbo = karbohidrat * 4
    kalori_protein = protein * 4
    total_kalori_makro = kalori_lemak + kalori_karbo + kalori_protein

    if total_kalori_makro > 0:
        df_makro = pd.DataFrame({
            "Sumber": ["Lemak (9 kkal/g)", "Karbohidrat (4 kkal/g)", "Protein (4 kkal/g)"],
            "Kalori": [kalori_lemak, kalori_karbo, kalori_protein],
            "Gram": [lemak_total, karbohidrat, protein]
        })

        fig_makro = px.pie(
            df_makro,
            values="Kalori",
            names="Sumber",
            hole=0.45,
            color="Sumber",
            color_discrete_map={
                "Lemak (9 kkal/g)": "#EF553B",
                "Karbohidrat (4 kkal/g)": "#00CC96",
                "Protein (4 kkal/g)": "#636EFA"
            },
            title="Distribusi Sumber Kalori (Macronutrient Split)",
            hover_data=['Gram']
        )
        fig_makro.update_traces(textposition='inside', textinfo='percent+label')
        fig_makro.update_layout(height=350, margin=dict(t=40, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig_makro, use_container_width=True)

    # 3. Row Bawah: Progress Bar AKG (Angka Kecukupan Gizi) per Takaran Saji
    st.markdown("#### Pemenuhan Angka Kecukupan Gizi (AKG) Harian")
    st.write(f"Persentase batas harian profil **{user_profile}** yang terpakai untuk **1 Takaran Saji ({takaran_saji}g/ml)** produk ini:")

    # Gula
    pct_gula = (gula / current_threshold['gula']) * 100 if current_threshold['gula'] > 0 else 0
    st.write(f"**Gula**: {gula}g dari batas {current_threshold['gula']}g/hari")
    st.progress(min(int(pct_gula), 100))
    if pct_gula > 50:
        st.warning(f"⚠️ 1 Porsi produk ini menghabiskan **{pct_gula:.1f}%** jatah gula harian Anda!")

    # Natrium
    pct_natrium = (natrium / current_threshold['natrium']) * 100 if current_threshold['natrium'] > 0 else 0
    st.write(f"**Natrium**: {natrium}mg dari batas {current_threshold['natrium']}mg/hari")
    st.progress(min(int(pct_natrium), 100))
    if pct_natrium > 50:
        st.warning(f"⚠️ 1 Porsi produk ini menghabiskan **{pct_natrium:.1f}%** jatah natrium harian Anda!")

    # Lemak Jenuh
    pct_lemak_jenuh = (lemak_jenuh / current_threshold['lemak_jenuh']) * 100 if current_threshold['lemak_jenuh'] > 0 else 0
    st.write(f"**Lemak Jenuh**: {lemak_jenuh}g dari batas {current_threshold['lemak_jenuh']}g/hari")
    st.progress(min(int(pct_lemak_jenuh), 100))
    if pct_lemak_jenuh > 50:
        st.warning(f"⚠️ 1 Porsi produk ini menghabiskan **{pct_lemak_jenuh:.1f}%** jatah lemak jenuh harian Anda!")


# --- FUNGSI HELPER UNTUK EXPORT HTML REPORT ---
def generate_html_report(product_name, risk_score, recommendation, upf_ingredients, nutrition_data, tdee_profile, kepadatan_energi, takaran_saji):
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    upf_status = "YA (Mengandung Aditif Sintetik)" if len(upf_ingredients) > 0 else "TIDAK (Relatif Alami)"
    upf_details = ", ".join(upf_ingredients) if len(upf_ingredients) > 0 else "-"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <title>Health Report: {product_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
            .header {{ text-align: center; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
            .section {{ margin-top: 20px; }}
            .risk-high {{ color: #D32F2F; font-weight: bold; }}
            .risk-med {{ color: #F57C00; font-weight: bold; }}
            .risk-low {{ color: #388E3C; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .footer {{ margin-top: 40px; font-size: 12px; text-align: center; color: #777; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h2>SMART NutriScan AI - Executive Health Report</h2>
            <p>Tanggal Cetak: {now_str}</p>
        </div>

        <div class="section">
            <h3>Informasi Produk</h3>
            <p><strong>Nama Produk:</strong> {product_name}</p>
            <p><strong>Takaran Saji Analisis:</strong> {takaran_saji} g/ml</p>
        </div>

        <div class="section">
            <h3>Hasil Prediksi AI</h3>
            <p><strong>Skor Risiko Machine Learning:</strong> {risk_score:.2f}%</p>
            <p><strong>Rekomendasi Konsumsi:</strong> {recommendation}</p>
            <p><strong>Status Ultra-Processed Food (UPF):</strong> {upf_status}</p>
            <p><strong>Aditif Terdeteksi (NLP):</strong> {upf_details}</p>
        </div>

        <div class="section">
            <h3>Business Intelligence & Health Metrics</h3>
            <p><strong>Profil Kebutuhan Pengguna:</strong> Kalori {tdee_profile['kalori']:.0f} kkal | Gula {tdee_profile['gula']:.1f} g | Natrium {tdee_profile['natrium']} mg</p>
            <p><strong>Kepadatan Energi:</strong> {kepadatan_energi:.2f} kkal/g</p>

            <table>
                <tr>
                    <th>Nutrisi</th>
                    <th>Kandungan per Saji</th>
                    <th>% Pemenuhan Batas Harian Profil</th>
                </tr>
                <tr>
                    <td>Gula</td>
                    <td>{nutrition_data.get('gula', 0)} g</td>
                    <td>{((nutrition_data.get('gula', 0) / tdee_profile['gula']) * 100):.1f}%</td>
                </tr>
                <tr>
                    <td>Natrium</td>
                    <td>{nutrition_data.get('natrium', 0)} mg</td>
                    <td>{((nutrition_data.get('natrium', 0) / tdee_profile['natrium']) * 100):.1f}%</td>
                </tr>
                <tr>
                    <td>Lemak Jenuh</td>
                    <td>{nutrition_data.get('lemak_jenuh', 0)} g</td>
                    <td>{((nutrition_data.get('lemak_jenuh', 0) / tdee_profile['lemak_jenuh']) * 100):.1f}%</td>
                </tr>
            </table>
        </div>

        <div class="footer">
            <p>Laporan ini digenerate secara otomatis oleh model Hybrid CBLIGHT-WOA & BI Analytics.</p>
        </div>
    </body>
    </html>
    """
    return html_content


# --- FUNGSI HELPER UNTUK DETEKSI NLP UPF ---
def deteksi_upf_nlp(komposisi_text):
    """
    Fungsi NLP/Regex untuk mendeteksi bahan kimia yang mengindikasikan
    produk adalah Ultra-Processed Food (UPF).
    """
    upf_keywords = [
        "aspartam", "sukralosa", "sirup fruktosa", "fruktosa sirup", "maltodekstrin",
        "perisa sintetik", "pewarna sintetik", "tartrazin", "karmoisin", "eritrosin",
        "pengawet", "natrium benzoat", "kalium sorbat", "penguat rasa", "msg",
        "mononatrium glutamat", "tbhq", "bht", "pengembang sintetik", "pemanis buatan"
    ]

    found_ingredients = []
    text_lower = str(komposisi_text).lower()

    for kw in upf_keywords:
        if kw in text_lower:
            found_ingredients.append(kw.title())

    return found_ingredients

# --- FUNGSI HELPER UNTUK KALKULASI TDEE ---
def hitung_tdee_dinamis(gender, usia, berat, tinggi, aktivitas):
    # Rumus BMR Mifflin-St Jeor
    if gender == "Pria":
        bmr = (10 * berat) + (6.25 * tinggi) - (5 * usia) + 5
    else:
        bmr = (10 * berat) + (6.25 * tinggi) - (5 * usia) - 161

    # Faktor Aktivitas Fisik
    faktor = {
        "Sedentary (Jarang Olahraga)": 1.2,
        "Ringan (Olahraga 1-3x/minggu)": 1.375,
        "Sedang (Olahraga 3-5x/minggu)": 1.55,
        "Aktif (Olahraga 6-7x/minggu)": 1.725,
        "Sangat Aktif (Pekerja Fisik / Atlet)": 1.9
    }

    tdee = bmr * faktor.get(aktivitas, 1.2)

    # Perhitungan Threshold Gizi (Standar WHO)
    # Gula: Max 10% dari total kalori (1g gula = 4 kkal)
    max_gula_g = (tdee * 0.10) / 4

    # Lemak Jenuh: Max 10% dari total kalori (1g lemak = 9 kkal)
    max_lemak_jenuh_g = (tdee * 0.10) / 9

    return {
        "kalori": tdee,
        "gula": max_gula_g,
        "lemak_jenuh": max_lemak_jenuh_g,
        "natrium": 2000 # Standar natrium harian umumnya flat 2000mg untuk orang sehat
    }


# --- UI Aplikasi ---

# --- Sidebar ---
with st.sidebar:
    st.image("assets/Logo Smart NutriScan AI.png", width=150)
    st.title("SMART NutriScan AI")
    
    st.header("⚙️ Profil Personal & BMR")

    # Advanced Personalization (BMR & TDEE)
    st.markdown("Kustomisasi batas asupan harian Anda.")

    col_g, col_u = st.columns(2)
    with col_g:
        user_gender = st.selectbox("Gender", ["Pria", "Wanita"])
    with col_u:
        user_age = st.number_input("Usia (Tahun)", min_value=1, max_value=120, value=25)

    col_w, col_h = st.columns(2)
    with col_w:
        user_weight = st.number_input("Berat (kg)", min_value=10.0, max_value=300.0, value=65.0)
    with col_h:
        user_height = st.number_input("Tinggi (cm)", min_value=50.0, max_value=250.0, value=165.0)

    user_activity = st.selectbox(
        "Tingkat Aktivitas",
        ["Sedentary (Jarang Olahraga)", "Ringan (Olahraga 1-3x/minggu)", "Sedang (Olahraga 3-5x/minggu)", "Aktif (Olahraga 6-7x/minggu)", "Sangat Aktif (Pekerja Fisik / Atlet)"]
    )
    
    # Kondisi Medis Khusus (Bisa override perhitungan TDEE standar)
    kondisi_medis = st.selectbox("Kondisi Khusus (Opsional)", ["Tidak Ada", "Penderita Hipertensi", "Risiko Penyakit Ginjal", "Anak-anak (Pre-set)"])

    # Kalkulasi
    calculated_threshold = hitung_tdee_dinamis(user_gender, user_age, user_weight, user_height, user_activity)

    # Terapkan Override jika ada kondisi khusus
    if kondisi_medis == "Penderita Hipertensi":
        calculated_threshold["natrium"] = 1200
    elif kondisi_medis == "Risiko Penyakit Ginjal":
        calculated_threshold["natrium"] = 1000
        calculated_threshold["kalori"] = calculated_threshold["kalori"] * 0.9 # Penyesuaian umum
    elif kondisi_medis == "Anak-anak (Pre-set)":
        calculated_threshold["gula"] = 25
        calculated_threshold["natrium"] = 1500

    # Set sebagai threshold yang aktif digunakan
    current_threshold = calculated_threshold

    with st.expander("Lihat Kebutuhan Harian Anda (AKG)", expanded=False):
        st.write(f"**Kalori (TDEE):** {current_threshold['kalori']:.0f} kkal")
        st.write(f"**Max Gula:** {current_threshold['gula']:.1f} g")
        st.write(f"**Max Lemak Jenuh:** {current_threshold['lemak_jenuh']:.1f} g")
        st.write(f"**Max Natrium:** {current_threshold['natrium']} mg")

    st.markdown("---")
    
    app_mode = st.radio(
        "Pilih Fitur:",
        ["Analisis Produk Tunggal", "Scan from Image", "Analisis Batch (Excel)", "Perbandingan Produk", "Simulasi Konsumsi", "Riwayat Analisis", "Edukasi Gizi"]
    )
    st.markdown("---")
    st.info("Dashboard ini adalah sistem intelijen terpadu. Fitur Analisis AI di-back-up oleh model hybrid CBLIGHT-WOA & BI Analytics.")

# --- Halaman Utama ---

if app_mode == "Analisis Produk Tunggal":
    st.header("Analisis Produk Pangan dengan AI")

    if not all([feat_model, lgbm_model, w2v_model, scaler]):
        st.error("Model tidak dapat digunakan. Silakan periksa log kesalahan di konsol.")
    else:
        st.success("Model AI aktif dan siap digunakan untuk analisis.")
        st.markdown("---")

        main_col, right_col = st.columns([1.8, 1.2])

        with main_col:
            st.subheader("Input Informasi Produk")
            st.markdown("Isi form di bawah ini dengan informasi dari label nutrisi produk.")
            
            product_name = st.text_input("Nama Produk", "Biskuit Cokelat")

            # Form untuk input data (Ditambah Takaran Saji untuk BI)
            c0, c1, c2 = st.columns(3)
            takaran_saji = c0.number_input("Takaran Saji (g/ml)", min_value=1.0, value=30.0, format="%.1f")
            energi = c1.number_input("Energi (kkal)", min_value=0, value=180)
            lemak_total = c2.number_input("Lemak Total (g)", min_value=0.0, value=8.0, format="%.1f")

            c3, c4, c5 = st.columns(3)
            lemak_jenuh = c3.number_input("Lemak Jenuh (g)", min_value=0.0, value=4.0, format="%.1f")
            protein = c4.number_input("Protein (g)", min_value=0.0, value=2.0, format="%.1f")
            karbohidrat = c5.number_input("Karbohidrat (g)", min_value=0.0, value=25.0, format="%.1f")

            c6, c7, c8, c9 = st.columns(4)
            gula = c6.number_input("Gula (g)", min_value=0.0, value=15.0, format="%.1f")
            garam = c7.number_input("Garam (g)", min_value=0.0, value=0.3, format="%.2f")
            natrium = c8.number_input("Natrium (mg)", min_value=0, value=200)
            natrium_benzoat = c9.number_input("Natrium Benzoat (mg)", min_value=0.0, value=0.0, format="%.2f")

            komposisi = st.text_area("Komposisi / Ingredients", "Tepung Terigu, Gula, Minyak Nabati, Cokelat Bubuk, Pengembang, Perisa Sintetik, Garam.")

            analyze_button = st.button("✨ Analisis AI & Gizi Sekarang!", type="primary")

        with right_col:
            st.subheader("Hasil Analisis AI (Prediksi Risiko)")
            if analyze_button:
                with st.spinner('Menganalisis produk dengan model CBLIGHT-WOA...'):
                    # Data dictionary ini TETAP murni untuk model ML, tanpa diganggu metrik BI
                    nutrition_data = {
                        'energi': energi, 'lemak_total': lemak_total, 'lemak_jenuh': lemak_jenuh,
                        'protein': protein, 'karbohidrat': karbohidrat, 'gula': gula,
                        'garam': garam, 'natrium': natrium, 'natrium_benzoat': natrium_benzoat
                    }

                    risk_score, xai_factors, recommendation = analyze_product_fully(
                        nutrition_data, komposisi, feat_model, lgbm_model, w2v_model, scaler
                    )

                    from datetime import datetime
                    display_profile = kondisi_medis if kondisi_medis != "Tidak Ada" else f"{user_gender} {user_age} Thn"
                    st.session_state.scan_history.append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "product_name": product_name,
                        "risk_score": risk_score,
                        "profile": display_profile,
                        "nutrition": nutrition_data
                    })

                    st.metric(label="Skor Risiko Prediksi ML", value=f"{risk_score:.2f}%")
                    if risk_score > 75:
                        st.error("🔴 Risiko Sangat Tinggi")
                    elif risk_score > 50:
                        st.warning("🟠 Risiko Tinggi")
                    elif risk_score > 25:
                        st.warning("🟡 Risiko Sedang")
                    else:
                        st.success("🟢 Risiko Rendah")

                    st.markdown("---")
                    st.markdown("#### Radar Kontribusi Nutrisi (XAI)")

                    categories = list(xai_factors.keys())
                    values = list(xai_factors.values())

                    norm_values = []
                    for k, v in xai_factors.items():
                        if 'gula' in k.lower(): norm_values.append(min((v / 50) * 100, 100))
                        elif 'natrium' in k.lower() and 'benzoat' not in k.lower(): norm_values.append(min((v / 1500) * 100, 100))
                        elif 'lemak' in k.lower(): norm_values.append(min((v / 67) * 100, 100))
                        elif 'energi' in k.lower(): norm_values.append(min((v / 2000) * 100, 100))
                        else: norm_values.append(min((v / 100) * 100, 100))

                    fig_radar = go.Figure()

                    fig_radar.add_trace(go.Scatterpolar(
                        r=norm_values + [norm_values[0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        name='Kandungan Produk',
                        line_color='red' if risk_score > 50 else 'orange' if risk_score > 25 else 'green',
                        hovertemplate="Feature: %{theta}<br>Skor Relatif: %{r:.1f}/100<extra></extra>"
                    ))

                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
                        showlegend=False,
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=250
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                    st.markdown("#### Rekomendasi ML")
                    st.info(recommendation)

                    # --- FITUR NLP UPF ---
                    upf_ingredients = deteksi_upf_nlp(komposisi)
                    if len(upf_ingredients) > 0:
                        st.markdown("---")
                        st.error("🚨 **Peringatan: Makanan Ultra-Proses (UPF)!**")
                        st.write("Berdasarkan Analisis Teks (NLP), produk ini mengandung bahan aditif sintetik/industri berikut:")
                        st.write(", ".join([f"**{ing}**" for ing in upf_ingredients]))
                        st.caption("Konsumsi makanan ultra-proses yang rutin dikaitkan dengan risiko penyakit metabolik jangka panjang.")

            else:
                st.metric(label="Skor Risiko Prediksi", value="-")
                st.info("Input data dan jalankan analisis untuk melihat detail prediksi risiko, Radar XAI, dan Rekomendasi ML.")

        # Eksekusi visualisasi Business & Health Metrics di bawah setelah tombol di-klik
        if analyze_button:
            display_profile = kondisi_medis if kondisi_medis != "Tidak Ada" else f"{user_gender} {user_age} Thn"
            render_holistic_nutrition_metrics(energi, takaran_saji, lemak_total, karbohidrat, protein, gula, natrium, lemak_jenuh, current_threshold, display_profile)

            st.markdown("---")
            # Generate and Download HTML Report
            kepadatan_energi = energi / takaran_saji if takaran_saji > 0 else 0
            html_report = generate_html_report(product_name, risk_score, recommendation, upf_ingredients, nutrition_data, current_threshold, kepadatan_energi, takaran_saji)

            st.download_button(
                label="📥 Download Executive Health Report (HTML)",
                data=html_report,
                file_name=f"Health_Report_{product_name.replace(' ', '_')}.html",
                mime="text/html",
                type="primary"
            )


elif app_mode == "Scan from Image":
    st.header("Scan Produk Otomatis (Kamera / Galeri)")
    st.info("Karena letak Informasi Nilai Gizi dan Komposisi Bahan sering terpisah di kemasan, Anda bisa melakukan pemindaian (scan) dua kali menggunakan Kamera langsung atau unggah foto dari Galeri.")

    # Inisialisasi variabel parsed_data agar tidak error
    parsed_data = {
        'energi': 0, 'lemak_total': 0.0, 'lemak_jenuh': 0.0, 'protein': 0.0,
        'karbohidrat': 0.0, 'gula': 0.0, 'garam': 0.0, 'natrium': 0,
        'natrium_benzoat': 0.0, 'komposisi': '', 'product_name': ''
    }

    col_scan1, col_scan2 = st.columns(2)

    # ------------------ SCAN 1: NILAI GIZI ------------------
    with col_scan1:
        st.markdown("### 📸 Scan 1: Tabel Nilai Gizi")
        input_type_1 = st.radio("Metode Input Nilai Gizi:", ["Upload File", "Kamera Langsung"], key="radio_1")

        img_file_1 = None
        if input_type_1 == "Upload File":
            img_file_1 = st.file_uploader("Pilih foto Tabel Gizi...", type=["jpg", "jpeg", "png"], key="upload_1")
        else:
            img_file_1 = st.camera_input("Ambil foto Tabel Gizi langsung", key="cam_1")

        if img_file_1 is not None:
            image_1 = Image.open(img_file_1)
            with st.expander("Lihat Hasil Preprocessing OCR Tabel Gizi", expanded=False):
                with st.spinner("Mengekstraksi Tabel Gizi dengan Computer Vision..."):
                    enhanced_image_1 = preprocess_image_for_ocr(image_1)
                    st.image(enhanced_image_1, caption="Binarization & CLAHE (Gizi)", use_container_width=True)

                    img_byte_arr_1 = io.BytesIO()
                    enhanced_image_1.save(img_byte_arr_1, format='PNG')
                    ocr_results_1 = reader.readtext(img_byte_arr_1.getvalue(), detail=0, paragraph=False)

                    parsed_gizi = parse_nutrition_text(ocr_results_1, is_composition_only=False)
                    # Update global data dari parse 1
                    for k, v in parsed_gizi.items():
                        if v != 0 and v != "Tidak terdeteksi." and v != "Produk Tanpa Nama":
                            parsed_data[k] = v
                    st.success("Tabel Gizi berhasil diproses!")

    # ------------------ SCAN 2: KOMPOSISI ------------------
    with col_scan2:
        st.markdown("### 📸 Scan 2: Teks Komposisi (Opsional)")
        st.caption("Gunakan jika letak komposisi jauh dari tabel gizi.")
        input_type_2 = st.radio("Metode Input Komposisi:", ["Upload File", "Kamera Langsung"], key="radio_2")

        img_file_2 = None
        if input_type_2 == "Upload File":
            img_file_2 = st.file_uploader("Pilih foto Teks Komposisi...", type=["jpg", "jpeg", "png"], key="upload_2")
        else:
            img_file_2 = st.camera_input("Ambil foto Teks Komposisi langsung", key="cam_2")

        if img_file_2 is not None:
            image_2 = Image.open(img_file_2)
            with st.expander("Lihat Hasil Preprocessing OCR Komposisi", expanded=False):
                with st.spinner("Mengekstraksi Teks Komposisi dengan Computer Vision..."):
                    enhanced_image_2 = preprocess_image_for_ocr(image_2)
                    st.image(enhanced_image_2, caption="Binarization & CLAHE (Komposisi)", use_container_width=True)

                    img_byte_arr_2 = io.BytesIO()
                    enhanced_image_2.save(img_byte_arr_2, format='PNG')
                    ocr_results_2 = reader.readtext(img_byte_arr_2.getvalue(), detail=0, paragraph=False)

                    parsed_komposisi = parse_nutrition_text(ocr_results_2, is_composition_only=True)
                    if parsed_komposisi['komposisi'] != "Tidak terdeteksi.":
                        parsed_data['komposisi'] = parsed_komposisi['komposisi']
                    st.success("Teks Komposisi berhasil diproses!")

    st.markdown("---")

    if img_file_1 is not None or img_file_2 is not None:
        main_col, right_col = st.columns([1.8, 1.2])

        with main_col:
            st.subheader("📝 Konfirmasi Data Input (Hasil OCR)")
            st.info("Silakan lengkapi **Takaran Saji** dan koreksi jika ada salah baca.")
            
            product_name = st.text_input("Nama Produk", value=parsed_data.get('product_name', ''))

            # Tambahan input takaran saji karena OCR sering miss bagian ini jika formatnya aneh
            c0, c1, c2 = st.columns(3)
            takaran_saji = c0.number_input("Takaran Saji (g/ml)", min_value=1.0, value=30.0, format="%.1f", help="OCR sulit menangkap ini. Mohon isi manual.")
            energi = c1.number_input("Energi (kkal)", min_value=0, value=int(parsed_data.get('energi', 0)))
            lemak_total = c2.number_input("Lemak Total (g)", min_value=0.0, value=parsed_data.get('lemak_total', 0.0), format="%.1f")

            c3, c4, c5 = st.columns(3)
            lemak_jenuh = c3.number_input("Lemak Jenuh (g)", min_value=0.0, value=parsed_data.get('lemak_jenuh', 0.0), format="%.1f")
            protein = c4.number_input("Protein (g)", min_value=0.0, value=parsed_data.get('protein', 0.0), format="%.1f")
            karbohidrat = c5.number_input("Karbohidrat (g)", min_value=0.0, value=parsed_data.get('karbohidrat', 0.0), format="%.1f")

            c6, c7, c8, c9 = st.columns(4)
            gula = c6.number_input("Gula (g)", min_value=0.0, value=parsed_data.get('gula', 0.0), format="%.1f")
            garam = c7.number_input("Garam (g)", min_value=0.0, value=parsed_data.get('garam', 0.0), format="%.2f")
            natrium = c8.number_input("Natrium (mg)", min_value=0, value=int(parsed_data.get('natrium', 0)))
            natrium_benzoat = c9.number_input("Natrium Benzoat (mg)", min_value=0.0, value=parsed_data.get('natrium_benzoat', 0.0), format="%.2f")

            komposisi = st.text_area("Komposisi / Ingredients", value=parsed_data.get('komposisi', ''), height=100)

            analyze_button = st.button("✨ Analisis AI & Gizi Sekarang!", type="primary")

        with right_col:
            st.subheader("Hasil Analisis AI (Prediksi Risiko)")
            if analyze_button:
                with st.spinner('Menganalisis produk dengan model CBLIGHT-WOA...'):
                    nutrition_data = {
                        'energi': energi, 'lemak_total': lemak_total, 'lemak_jenuh': lemak_jenuh,
                        'protein': protein, 'karbohidrat': karbohidrat, 'gula': gula,
                        'garam': garam, 'natrium': natrium, 'natrium_benzoat': natrium_benzoat
                    }

                    risk_score, xai_factors, recommendation = analyze_product_fully(
                        nutrition_data, komposisi, feat_model, lgbm_model, w2v_model, scaler
                    )

                    from datetime import datetime
                    display_profile = kondisi_medis if kondisi_medis != "Tidak Ada" else f"{user_gender} {user_age} Thn"
                    st.session_state.scan_history.append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "product_name": product_name,
                        "risk_score": risk_score,
                        "profile": display_profile,
                        "nutrition": nutrition_data
                    })

                    st.metric(label="Skor Risiko Prediksi", value=f"{risk_score:.2f}%")
                    if risk_score > 75: st.error("🔴 Risiko Sangat Tinggi")
                    elif risk_score > 50: st.warning("🟠 Risiko Tinggi")
                    elif risk_score > 25: st.warning("🟡 Risiko Sedang")
                    else: st.success("🟢 Risiko Rendah")

                    st.markdown("---")
                    st.markdown("#### Radar Kontribusi Nutrisi (XAI)")

                    categories = list(xai_factors.keys())
                    values = list(xai_factors.values())

                    norm_values = []
                    for k, v in xai_factors.items():
                        if 'gula' in k.lower(): norm_values.append(min((v / 50) * 100, 100))
                        elif 'natrium' in k.lower() and 'benzoat' not in k.lower(): norm_values.append(min((v / 1500) * 100, 100))
                        elif 'lemak' in k.lower(): norm_values.append(min((v / 67) * 100, 100))
                        elif 'energi' in k.lower(): norm_values.append(min((v / 2000) * 100, 100))
                        else: norm_values.append(min((v / 100) * 100, 100))

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=norm_values + [norm_values[0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        name='Kandungan Produk',
                        line_color='red' if risk_score > 50 else 'orange' if risk_score > 25 else 'green',
                        hovertemplate="Feature: %{theta}<br>Skor Relatif: %{r:.1f}/100<extra></extra>"
                    ))

                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
                        showlegend=False, margin=dict(l=20, r=20, t=20, b=20), height=250
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                    st.markdown("#### Rekomendasi ML")
                    st.info(recommendation)

                    # --- FITUR NLP UPF ---
                    upf_ingredients = deteksi_upf_nlp(komposisi)
                    if len(upf_ingredients) > 0:
                        st.markdown("---")
                        st.error("🚨 **Peringatan: Makanan Ultra-Proses (UPF)!**")
                        st.write("Berdasarkan Analisis Teks (NLP), produk ini mengandung bahan aditif sintetik/industri berikut:")
                        st.write(", ".join([f"**{ing}**" for ing in upf_ingredients]))
                        st.caption("Konsumsi makanan ultra-proses yang rutin dikaitkan dengan risiko penyakit metabolik jangka panjang.")
            else:
                 st.metric(label="Skor Risiko Prediksi", value="-")
                 st.info("Jalankan analisis untuk melihat hasil AI.")

        # Panggil render module BI
        if analyze_button:
            display_profile = kondisi_medis if kondisi_medis != "Tidak Ada" else f"{user_gender} {user_age} Thn"
            render_holistic_nutrition_metrics(energi, takaran_saji, lemak_total, karbohidrat, protein, gula, natrium, lemak_jenuh, current_threshold, display_profile)

            st.markdown("---")
            # Generate and Download HTML Report
            kepadatan_energi = energi / takaran_saji if takaran_saji > 0 else 0
            html_report = generate_html_report(product_name, risk_score, recommendation, upf_ingredients, nutrition_data, current_threshold, kepadatan_energi, takaran_saji)

            st.download_button(
                label="📥 Download Executive Health Report (HTML)",
                data=html_report,
                file_name=f"Health_Report_OCR_{product_name.replace(' ', '_')}.html",
                mime="text/html",
                type="primary"
            )








elif app_mode == "Analisis Batch (Excel)":
    st.header("Analisis Batch Produk dari File Excel")
    st.write("Unggah file Excel dengan daftar produk dan informasi nutrisinya untuk dianalisis secara bersamaan.")
    
    expected_columns = [
        'Energi', 'Lemak', 'Karbohidrat', 'Gula', 'Protein', 'Garam', 'Komposisi'
    ]
    
    st.info(f"Pastikan file Excel Anda memiliki kolom: {', '.join(expected_columns)}")

    uploaded_excel = st.file_uploader("Pilih file .xlsx", type=["xlsx"])
    
    if uploaded_excel:
        df = pd.read_excel(uploaded_excel)
        
        # Preprocess the batch data to clean units and handle decimal separators
        df = preprocess_batch_excel_data(df)
        
        st.dataframe(df)
        
        # Verify columns exist
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            st.error(f"File Excel tidak memiliki kolom yang dibutuhkan: {', '.join(missing_cols)}")
        else:
            batch_analyze_button = st.button("Mulai Analisis Batch", type="primary", disabled=not all([feat_model, lgbm_model, w2v_model, scaler]))
            
            if batch_analyze_button:
                results = []
                total_rows = len(df)
                progress_bar = st.progress(0)
                
                with st.spinner(f"Menganalisis {total_rows} produk..."):
                    for i, row in df.iterrows():
                        nutrition_data = {
                            'energi': float(row.get('Energi', 0)),
                            'lemak_total': float(row.get('Lemak', 0)),
                            'karbohidrat': float(row.get('Karbohidrat', 0)),
                            'gula': float(row.get('Gula', 0)),
                            'protein': float(row.get('Protein', 0)),
                            'garam': float(row.get('Garam', 0)),
                            'natrium_benzoat': float(row.get('Natrium Benzoat', 0))
                        }
                        composition_text = row.get('Komposisi', "")

                        risk_score, _, _ = analyze_product_fully(
                            nutrition_data, composition_text, feat_model, lgbm_model, w2v_model, scaler
                        )
                        results.append(risk_score)
                        progress_bar.progress((i + 1) / total_rows)
                
                st.success(f"Analisis batch selesai untuk {total_rows} produk!")
                
                df_results = df.copy()
                df_results['Risk Score (%)'] = [f"{r:.2f}" for r in results]
                
                st.subheader("Hasil Analisis Batch")
                st.dataframe(df_results)
                
                df_xlsx = to_excel(df_results)
                st.download_button(
                    label="📥 Download Hasil Analisis (.xlsx)",
                    data=df_xlsx,
                    file_name="hasil_analisis_batch.xlsx"
                )

elif app_mode == "Perbandingan Produk":
    st.header("Perbandingan Produk (Food Comparison Mode)")
    st.info("Bandingkan metrik AI (Skor Risiko) dan metrik BI (Kepadatan Energi) dari dua produk sekaligus.")

    # Inisialisasi session state untuk data produk jika belum ada
    if 'product_a_data' not in st.session_state:
        st.session_state.product_a_data = {
            'name': 'Sereal Pagi A', 'takaran': 30.0, 'energi': 150, 'lemak_total': 5.0,
            'lemak_jenuh': 1.0, 'protein': 3.0, 'karbohidrat': 30.0, 'gula': 12.0,
            'natrium': 180, 'natrium_benzoat': 0.0, 'komposisi': 'Gandum Utuh, Gula, Garam.'
        }
    if 'product_b_data' not in st.session_state:
        st.session_state.product_b_data = {
            'name': 'Sereal Pagi B', 'takaran': 30.0, 'energi': 160, 'lemak_total': 6.0,
            'lemak_jenuh': 3.0, 'protein': 2.0, 'karbohidrat': 28.0, 'gula': 18.0,
            'natrium': 250, 'natrium_benzoat': 0.0, 'komposisi': 'Jagung, Gula, Sirup Fruktosa, Garam.'
        }
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Produk A")
        input_method_a = st.radio("Metode Input Produk A:", ["Input Manual", "Scan dengan Kamera"], key="metode_a")

        if input_method_a == "Scan dengan Kamera":
            img_file_a = st.camera_input("Ambil Foto Produk A", key="cam_a")
            if img_file_a:
                image_a = Image.open(img_file_a)
                with st.spinner("Memproses gambar Produk A..."):
                    enhanced_image_a = preprocess_image_for_ocr(image_a)
                    ocr_results_a = reader.readtext(np.array(enhanced_image_a), detail=0, paragraph=False)
                    parsed_data_a = parse_nutrition_text(ocr_results_a)
                    
                    # Update session state dengan data OCR, jaga nilai yang ada jika OCR gagal
                    st.session_state.product_a_data['name'] = parsed_data_a.get('product_name') or st.session_state.product_a_data['name']
                    st.session_state.product_a_data['energi'] = int(parsed_data_a.get('energi') or st.session_state.product_a_data['energi'])
                    st.session_state.product_a_data['lemak_total'] = parsed_data_a.get('lemak_total') or st.session_state.product_a_data['lemak_total']
                    st.session_state.product_a_data['lemak_jenuh'] = parsed_data_a.get('lemak_jenuh') or st.session_state.product_a_data['lemak_jenuh']
                    st.session_state.product_a_data['protein'] = parsed_data_a.get('protein') or st.session_state.product_a_data['protein']
                    st.session_state.product_a_data['karbohidrat'] = parsed_data_a.get('karbohidrat') or st.session_state.product_a_data['karbohidrat']
                    st.session_state.product_a_data['gula'] = parsed_data_a.get('gula') or st.session_state.product_a_data['gula']
                    st.session_state.product_a_data['natrium'] = int(parsed_data_a.get('natrium') or st.session_state.product_a_data['natrium'])
                    st.session_state.product_a_data['komposisi'] = parsed_data_a.get('komposisi') or st.session_state.product_a_data['komposisi']
                    st.success("Scan Produk A selesai!")

        # Gunakan data dari session state untuk semua input
        data_a = st.session_state.product_a_data
        p_a_name = st.text_input("Nama Produk A", value=data_a['name'], key="name_a")
        p_a_takaran = st.number_input("Takaran Saji A (g)", min_value=1.0, value=data_a['takaran'], format="%.1f", key="takaran_a")
        p_a_energi = st.number_input("Energi A (kkal)", min_value=0, value=data_a['energi'], key="energi_a")
        p_a_lemak_total = st.number_input("Lemak Total A (g)", min_value=0.0, value=data_a['lemak_total'], format="%.1f", key="lemak_a")
        p_a_lemak_jenuh = st.number_input("Lemak Jenuh A (g)", min_value=0.0, value=data_a['lemak_jenuh'], format="%.1f", key="lemak_jenuh_a")
        p_a_protein = st.number_input("Protein A (g)", min_value=0.0, value=data_a['protein'], format="%.1f", key="protein_a")
        p_a_karbohidrat = st.number_input("Karbohidrat A (g)", min_value=0.0, value=data_a['karbohidrat'], format="%.1f", key="karbo_a")
        p_a_gula = st.number_input("Gula A (g)", min_value=0.0, value=data_a['gula'], format="%.1f", key="gula_a")
        p_a_natrium = st.number_input("Natrium A (mg)", min_value=0, value=data_a['natrium'], key="natrium_a")
        p_a_natrium_benzoat = st.number_input("Natrium Benzoat A (mg)", min_value=0.0, value=data_a['natrium_benzoat'], format="%.2f", key="benzoat_a")
        p_a_komposisi = st.text_area("Komposisi A", value=data_a['komposisi'], height=100, key="komposisi_a")
        p_a_garam = p_a_natrium / 400

    with col2:
        st.subheader("Produk B")
        input_method_b = st.radio("Metode Input Produk B:", ["Input Manual", "Scan dengan Kamera"], key="metode_b")

        if input_method_b == "Scan dengan Kamera":
            img_file_b = st.camera_input("Ambil Foto Produk B", key="cam_b")
            if img_file_b:
                image_b = Image.open(img_file_b)
                with st.spinner("Memproses gambar Produk B..."):
                    enhanced_image_b = preprocess_image_for_ocr(image_b)
                    ocr_results_b = reader.readtext(np.array(enhanced_image_b), detail=0, paragraph=False)
                    parsed_data_b = parse_nutrition_text(ocr_results_b)

                    st.session_state.product_b_data['name'] = parsed_data_b.get('product_name') or st.session_state.product_b_data['name']
                    st.session_state.product_b_data['energi'] = int(parsed_data_b.get('energi') or st.session_state.product_b_data['energi'])
                    st.session_state.product_b_data['lemak_total'] = parsed_data_b.get('lemak_total') or st.session_state.product_b_data['lemak_total']
                    st.session_state.product_b_data['lemak_jenuh'] = parsed_data_b.get('lemak_jenuh') or st.session_state.product_b_data['lemak_jenuh']
                    st.session_state.product_b_data['protein'] = parsed_data_b.get('protein') or st.session_state.product_b_data['protein']
                    st.session_state.product_b_data['karbohidrat'] = parsed_data_b.get('karbohidrat') or st.session_state.product_b_data['karbohidrat']
                    st.session_state.product_b_data['gula'] = parsed_data_b.get('gula') or st.session_state.product_b_data['gula']
                    st.session_state.product_b_data['natrium'] = int(parsed_data_b.get('natrium') or st.session_state.product_b_data['natrium'])
                    st.session_state.product_b_data['komposisi'] = parsed_data_b.get('komposisi') or st.session_state.product_b_data['komposisi']
                    st.success("Scan Produk B selesai!")

        data_b = st.session_state.product_b_data
        p_b_name = st.text_input("Nama Produk B", value=data_b['name'], key="name_b")
        p_b_takaran = st.number_input("Takaran Saji B (g)", min_value=1.0, value=data_b['takaran'], format="%.1f", key="takaran_b")
        p_b_energi = st.number_input("Energi B (kkal)", min_value=0, value=data_b['energi'], key="energi_b")
        p_b_lemak_total = st.number_input("Lemak Total B (g)", min_value=0.0, value=data_b['lemak_total'], format="%.1f", key="lemak_b")
        p_b_lemak_jenuh = st.number_input("Lemak Jenuh B (g)", min_value=0.0, value=data_b['lemak_jenuh'], format="%.1f", key="lemak_jenuh_b")
        p_b_protein = st.number_input("Protein B (g)", min_value=0.0, value=data_b['protein'], format="%.1f", key="protein_b")
        p_b_karbohidrat = st.number_input("Karbohidrat B (g)", min_value=0.0, value=data_b['karbohidrat'], format="%.1f", key="karbo_b")
        p_b_gula = st.number_input("Gula B (g)", min_value=0.0, value=data_b['gula'], format="%.1f", key="gula_b")
        p_b_natrium = st.number_input("Natrium B (mg)", min_value=0, value=data_b['natrium'], key="natrium_b")
        p_b_natrium_benzoat = st.number_input("Natrium Benzoat B (mg)", min_value=0.0, value=data_b['natrium_benzoat'], format="%.2f", key="benzoat_b")
        p_b_komposisi = st.text_area("Komposisi B", value=data_b['komposisi'], height=100, key="komposisi_b")
        p_b_garam = p_b_natrium / 400

    st.markdown("---")
    compare_button = st.button("⚖️ Bandingkan Sekarang!", type="primary")

    if compare_button:
        # Update session state dari input manual sebelum analisis
        st.session_state.product_a_data.update({
            'name': p_a_name, 'takaran': p_a_takaran, 'energi': p_a_energi, 'lemak_total': p_a_lemak_total,
            'lemak_jenuh': p_a_lemak_jenuh, 'protein': p_a_protein, 'karbohidrat': p_a_karbohidrat,
            'gula': p_a_gula, 'natrium': p_a_natrium, 'natrium_benzoat': p_a_natrium_benzoat, 'komposisi': p_a_komposisi
        })
        st.session_state.product_b_data.update({
            'name': p_b_name, 'takaran': p_b_takaran, 'energi': p_b_energi, 'lemak_total': p_b_lemak_total,
            'lemak_jenuh': p_b_lemak_jenuh, 'protein': p_b_protein, 'karbohidrat': p_b_karbohidrat,
            'gula': p_b_gula, 'natrium': p_b_natrium, 'natrium_benzoat': p_b_natrium_benzoat, 'komposisi': p_b_komposisi
        })

        with st.spinner("Menganalisis dan membandingkan kedua produk..."):
            nutrition_a = {
                'energi': p_a_energi, 'lemak_total': p_a_lemak_total, 'lemak_jenuh': p_a_lemak_jenuh,
                'protein': p_a_protein, 'karbohidrat': p_a_karbohidrat, 'gula': p_a_gula,
                'garam': p_a_garam, 'natrium': p_a_natrium, 'natrium_benzoat': p_a_natrium_benzoat
            }
            nutrition_b = {
                'energi': p_b_energi, 'lemak_total': p_b_lemak_total, 'lemak_jenuh': p_b_lemak_jenuh,
                'protein': p_b_protein, 'karbohidrat': p_b_karbohidrat, 'gula': p_b_gula,
                'garam': p_b_garam, 'natrium': p_b_natrium, 'natrium_benzoat': p_b_natrium_benzoat
            }

            risk_a, _, _ = analyze_product_fully(nutrition_a, p_a_komposisi, feat_model, lgbm_model, w2v_model, scaler)
            risk_b, _, _ = analyze_product_fully(nutrition_b, p_b_komposisi, feat_model, lgbm_model, w2v_model, scaler)

            st.subheader("Pemenang Analisis Keseluruhan")
            
            # Kalkulasi Kepadatan Energi untuk BI Metric
            kepadatan_a = p_a_energi / p_a_takaran if p_a_takaran > 0 else 0
            kepadatan_b = p_b_energi / p_b_takaran if p_b_takaran > 0 else 0

            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown(f"### {p_a_name}")
                st.metric(label="Skor Risiko AI", value=f"{risk_a:.2f}%")
                st.metric(label="Kepadatan Energi", value=f"{kepadatan_a:.2f} kkal/g")

            with res_col2:
                st.markdown(f"### {p_b_name}")
                st.metric(label="Skor Risiko AI", value=f"{risk_b:.2f}%")
                st.metric(label="Kepadatan Energi", value=f"{kepadatan_b:.2f} kkal/g")

            st.markdown("---")

            if risk_a < risk_b:
                st.success(f"🏆 **{p_a_name}** adalah pilihan yang lebih baik secara algoritma AI dengan skor risiko lebih rendah.")
            elif risk_b < risk_a:
                st.success(f"🏆 **{p_b_name}** adalah pilihan yang lebih baik secara algoritma AI dengan skor risiko lebih rendah.")
            else:
                st.info("Kedua produk memiliki skor risiko yang sama.")

            st.subheader("Visualisasi Perbandingan Nutrisi (Normalisasi per 100g)")
            # Mengubah ke format per 100g agar perbandingannya apple-to-apple dalam grafik
            faktor_a = 100 / p_a_takaran if p_a_takaran > 0 else 0
            faktor_b = 100 / p_b_takaran if p_b_takaran > 0 else 0

            df_compare = pd.DataFrame({
                "Nutrisi (per 100g)": ["Gula (g)", "Natrium (mg)", "Lemak Jenuh (g)", "Karbohidrat (g)", "Protein (g)"],
                p_a_name: [p_a_gula * faktor_a, p_a_natrium * faktor_a, p_a_lemak_jenuh * faktor_a, p_a_karbohidrat * faktor_a, p_a_protein * faktor_a],
                p_b_name: [p_b_gula * faktor_b, p_b_natrium * faktor_b, p_b_lemak_jenuh * faktor_b, p_b_karbohidrat * faktor_b, p_b_protein * faktor_b]
            })

            df_melted = df_compare.melt(id_vars=["Nutrisi (per 100g)"], var_name="Produk", value_name="Kandungan")

            fig_compare_bar = px.bar(
                df_melted, x="Nutrisi (per 100g)", y="Kandungan", color="Produk", barmode="group",
                title=f"Perbandingan Nutrisi: {p_a_name} vs {p_b_name} (Distandarisasi per 100g)",
                text_auto='.1f'
            )
            fig_compare_bar.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_compare_bar, use_container_width=True)



elif app_mode == "Riwayat Analisis":
    st.header("Riwayat dan Monitoring Konsumsi")
    st.info("Berikut adalah riwayat analisis produk yang pernah Anda periksa.")

    if len(st.session_state.scan_history) == 0:
        st.write("Belum ada riwayat analisis.")
    else:
        history_df = pd.DataFrame(st.session_state.scan_history)

        # Categorize risk scores for visualization
        def categorize_risk(score):
            if score > 75: return "Sangat Tinggi"
            elif score > 50: return "Tinggi"
            elif score > 25: return "Sedang"
            else: return "Rendah"

        history_df["Kategori Risiko"] = history_df["risk_score"].apply(categorize_risk)

        # 1. Tabel Riwayat
        # --- FITUR BARU: HEALTH GRADE & CALENDAR HEATMAP ---
        st.subheader("Health Grade & Ringkasan Performa")

        avg_score = history_df["risk_score"].mean()
        total_scan = len(history_df)
        high_risk_count = len(history_df[history_df["risk_score"] > 50])

        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            # Kalkulasi Grade Kesehatan
            if avg_score < 25: grade, color = "A (Sangat Baik)", "normal"
            elif avg_score < 50: grade, color = "B (Cukup Baik)", "off"
            elif avg_score < 75: grade, color = "C (Berisiko)", "inverse"
            else: grade, color = "D (Bahaya)", "inverse"

            st.metric("Rata-rata Skor Kesehatan", f"{avg_score:.1f}%", delta=f"Grade: {grade}", delta_color=color)

        with col_g2:
            st.metric("Total Produk Dianalisis", str(total_scan))

        with col_g3:
            st.metric("Produk Berisiko Tinggi Ditemukan", str(high_risk_count), delta="-Kurangi konsumsi" if high_risk_count > 0 else "Aman", delta_color="inverse" if high_risk_count > 0 else "normal")

        st.markdown("---")

        # Calendar Heatmap Sederhana menggunakan Heatmap Plotly
        st.subheader("📅 Calendar Heatmap: Aktivitas Pengecekan Nutrisi")

        # Ekstrak tanggal saja tanpa waktu
        history_df["day_date"] = pd.to_datetime(history_df["date"]).dt.date

        # Agregasi jumlah scan dan rata-rata skor per hari
        heatmap_data = history_df.groupby("day_date").agg(
            total_scans=("product_name", "count"),
            avg_score=("risk_score", "mean")
        ).reset_index()

        # Mengubah tanggal ke string agar plotly mudah membacanya
        heatmap_data["day_date"] = heatmap_data["day_date"].astype(str)

        fig_heat = px.density_heatmap(
            heatmap_data,
            x="day_date",
            y="total_scans",
            z="avg_score",
            color_continuous_scale="RdYlGn_r", # Green is low risk, Red is high
            title="Intensitas Pengecekan Harian (Warna = Tingkat Risiko Rata-rata)",
            labels={"day_date": "Tanggal", "total_scans": "Frekuensi Produk Di-Scan", "avg_score": "Skor Risiko"}
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Kotak yang berwarna hijau berarti produk yang dikonsumsi rata-rata memiliki risiko rendah. Tetap jaga warnanya agar tidak merah!")

        st.markdown("---")

        st.subheader("Data Riwayat Lengkap")
        st.dataframe(history_df[["date", "product_name", "risk_score", "Kategori Risiko", "profile"]].style.format({"risk_score": "{:.2f}%"}))

        st.markdown("---")

        # 2. Visualisasi Dashboard
        st.subheader("Grafik Tren dan Proporsi")

        col1, col2 = st.columns(2)

        with col1:
            # Pie Chart - Proporsi Kategori Risiko
            fig_pie = px.pie(
                history_df,
                names="Kategori Risiko",
                title="Proporsi Kategori Risiko Produk",
                color="Kategori Risiko",
                color_discrete_map={
                    "Sangat Tinggi": "darkred",
                    "Tinggi": "red",
                    "Sedang": "orange",
                    "Rendah": "green"
                },
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Line Chart - Tren Skor Risiko
            fig_line = px.line(
                history_df,
                x="date",
                y="risk_score",
                title="Tren Skor Risiko Konsumsi Seiring Waktu",
                markers=True,
                hover_data=["product_name"]
            )
            fig_line.add_hrect(y0=0, y1=25, line_width=0, fillcolor="green", opacity=0.1)
            fig_line.add_hrect(y0=25, y1=50, line_width=0, fillcolor="orange", opacity=0.1)
            fig_line.add_hrect(y0=50, y1=75, line_width=0, fillcolor="red", opacity=0.1)
            fig_line.add_hrect(y0=75, y1=100, line_width=0, fillcolor="darkred", opacity=0.1)
            fig_line.update_yaxes(title="Skor Risiko (%)", range=[0, 100])
            st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("---")

        # 3. Scatter Plot Kompleks - Korelasi Gula dan Natrium vs Risiko
        st.subheader("Korelasi Kandungan Nutrisi Utama dan Risiko")
        st.info("Visualisasi ini memetakan kadar Gula dan Natrium produk yang pernah Anda periksa. Ukuran gelembung mewakili Skor Risiko.")

        # Ekstrak data nutrisi untuk plotting
        sugar_data = [item.get("gula", 0) for item in history_df["nutrition"]]
        sodium_data = [item.get("natrium", 0) for item in history_df["nutrition"]]

        scatter_df = pd.DataFrame({
            "Produk": history_df["product_name"],
            "Gula (g)": sugar_data,
            "Natrium (mg)": sodium_data,
            "Skor Risiko": history_df["risk_score"],
            "Kategori": history_df["Kategori Risiko"]
        })

        fig_scatter = px.scatter(
            scatter_df,
            x="Gula (g)",
            y="Natrium (mg)",
            size="Skor Risiko",
            color="Kategori",
            hover_name="Produk",
            title="Peta Risiko Berdasarkan Kandungan Gula dan Natrium",
            size_max=30,
            color_discrete_map={
                "Sangat Tinggi": "darkred",
                "Tinggi": "red",
                "Sedang": "orange",
                "Rendah": "green"
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("---")
        if st.button("Hapus Riwayat", type="secondary"):
            st.session_state.scan_history = []
            st.rerun()


elif app_mode == "Edukasi Gizi":
    st.header("Edukasi dan Rekomendasi Nutrisi Cerdas")

    st.markdown("### Batas Konsumsi Gizi Harian (Kemenkes RI)")
    st.info("Pedoman umum konsumsi gula, garam, dan lemak (G4G1L5) per hari untuk dewasa:")
    st.write("- **Gula:** 4 sendok makan (50 gram)")
    st.write("- **Garam:** 1 sendok teh (5 gram / 2000 mg Natrium)")
    st.write("- **Lemak:** 5 sendok makan (67 gram)")

    st.markdown("---")
    st.markdown("### Membaca Label Informasi Nilai Gizi")
    st.write("Perhatikan hal-hal berikut saat membaca label kemasan:")
    st.write("1. **Takaran Saji**: Semua nilai nutrisi yang tercantum biasanya berdasarkan satu takaran saji, bukan satu kemasan penuh.")
    st.write("2. **Kalori Total**: Perhatikan total kalori per sajian, terutama jika Anda sedang mengatur berat badan.")
    st.write("3. **Natrium/Garam**: Banyak produk camilan dan minuman kemasan menyembunyikan kadar natrium yang sangat tinggi.")

    st.markdown("---")
    st.markdown("### Alternatif Makanan Sehat")
    st.write("- **Ganti Minuman Manis**: Gunakan air putih, teh tawar, atau air infus buah.")
    st.write("- **Camilan Sehat**: Pilih buah potong, kacang edamame, atau yogurt tawar dibandingkan keripik kemasan.")
    st.write("- **Perbanyak Serat**: Konsumsi lebih banyak sayur dan biji-bijian utuh.")


elif app_mode == "Simulasi Konsumsi":
    st.header("Simulasi Konsumsi Produk")
    st.info("Masukkan detail produk dan perkirakan dampak risikonya berdasarkan frekuensi konsumsi Anda.")

    st.subheader("Langkah 1: Definisikan Produk")
    product_name = st.text_input("Nama Produk", "Minuman Soda")

    # Tambah input Takaran Saji untuk konsistensi meskipun tidak dipakai kalkulasi total serving di ML
    c0, c1, c2 = st.columns(3)
    takaran_saji = c0.number_input("Takaran Saji (g/ml)", min_value=1.0, value=250.0, format="%.1f")
    energi = c1.number_input("Energi (kkal)", min_value=0, value=150)
    lemak_total = c2.number_input("Lemak Total (g)", min_value=0.0, value=0.0, format="%.1f")

    c3, c4, c5 = st.columns(3)
    lemak_jenuh = c3.number_input("Lemak Jenuh (g)", min_value=0.0, value=0.0, format="%.1f")
    protein = c4.number_input("Protein (g)", min_value=0.0, value=0.0, format="%.1f")
    karbohidrat = c5.number_input("Karbohidrat (g)", min_value=0.0, value=40.0, format="%.1f")

    c6, c7, c8, c9 = st.columns(4)
    gula = c6.number_input("Gula (g)", min_value=0.0, value=39.0, format="%.1f")
    garam = c7.number_input("Garam (g)", min_value=0.0, value=0.1, format="%.2f")
    natrium = c8.number_input("Natrium (mg)", min_value=0, value=45)
    natrium_benzoat = c9.number_input("Natrium Benzoat (mg)", min_value=0.0, value=0.0, format="%.2f")

    komposisi = st.text_area("Komposisi / Ingredients", "Air Berkarbonasi, Gula, Sirup Fruktosa, Perisa Sintetik, Pengatur Keasaman.")

    st.markdown("---")
    st.subheader("Langkah 2: Atur Pola Konsumsi")
    freq_col, period_col = st.columns(2)
    with freq_col:
        frequency_per_week = st.number_input("Frekuensi konsumsi per minggu (kali/sajian)", min_value=1, value=3)
    with period_col:
        simulation_period_months = st.selectbox("Periode Simulasi (Bulan)", [1, 3, 6, 12])

    st.markdown("---")
    simulation_button = st.button("📈 Jalankan Simulasi", type="primary")

    if simulation_button:
        with st.spinner("Menjalankan simulasi konsumsi..."):
            nutrition_data = {
                'energi': energi, 'lemak_total': lemak_total, 'lemak_jenuh': lemak_jenuh,
                'protein': protein, 'karbohidrat': karbohidrat, 'gula': gula,
                'garam': garam, 'natrium': natrium, 'natrium_benzoat': natrium_benzoat
            }
            risk_score, _, _ = analyze_product_fully(
                nutrition_data, komposisi, feat_model, lgbm_model, w2v_model, scaler
            )

            st.subheader(f"Hasil Analisis Dasar untuk '{product_name}'")
            res_col, _ = st.columns(2)
            with res_col:
                st.metric(label="Skor Risiko per 1x Konsumsi", value=f"{risk_score:.2f}%")
                if risk_score > 75: st.error("Risiko Sangat Tinggi")
                elif risk_score > 50: st.warning("Risiko Tinggi")
                elif risk_score > 25: st.warning("Risiko Sedang")
                else: st.success("Risiko Rendah")

            st.markdown("---")
            st.subheader(f"Simulasi Akumulasi Selama {simulation_period_months} Bulan")

            profile_daily_limits = current_threshold

            days_in_period = simulation_period_months * 30.44
            weeks_in_period = days_in_period / 7
            total_servings = frequency_per_week * weeks_in_period

            total_gula = gula * total_servings
            total_natrium = natrium * total_servings
            total_lemak_jenuh = lemak_jenuh * total_servings

            limit_gula = profile_daily_limits['gula'] * days_in_period
            limit_natrium = profile_daily_limits['natrium'] * days_in_period
            limit_lemak_jenuh = profile_daily_limits['lemak_jenuh'] * days_in_period

            st.write(f"Dengan mengonsumsi **{product_name}** sebanyak **{frequency_per_week}** sajian seminggu, estimasi asupan Anda dari produk ini saja adalah:")

            percent_gula = (total_gula / limit_gula) * 100 if limit_gula > 0 else 0
            st.write(f"**Gula**: **{total_gula:.1f}g** / {limit_gula:.1f}g dari batas maksimal periode.")
            st.progress(min(int(percent_gula), 100))
            if percent_gula > 100:
                st.error(f"🔴 Peringatan! Konsumsi produk ini saja sudah **melebihi {percent_gula - 100:.0f}%** dari batas aman gula Anda.")
            elif percent_gula > 50:
                st.warning(f"🟡 Perhatian. Konsumsi produk ini menggunakan **{percent_gula:.0f}%** dari alokasi gula Anda untuk periode ini.")

            percent_natrium = (total_natrium / limit_natrium) * 100 if limit_natrium > 0 else 0
            st.write(f"**Natrium**: **{total_natrium / 1000:.2f}g** / {limit_natrium / 1000:.2f}g dari batas maksimal periode.")
            st.progress(min(int(percent_natrium), 100))
            if percent_natrium > 100:
                st.error(f"🔴 Peringatan! Konsumsi produk ini saja sudah **melebihi {percent_natrium - 100:.0f}%** dari batas aman natrium Anda.")
            elif percent_natrium > 50:
                st.warning(f"🟡 Perhatian. Konsumsi produk ini menggunakan **{percent_natrium:.0f}%** dari alokasi natrium Anda untuk periode ini.")

            percent_lemak_jenuh = (total_lemak_jenuh / limit_lemak_jenuh) * 100 if limit_lemak_jenuh > 0 else 0
            st.write(f"**Lemak Jenuh**: **{total_lemak_jenuh:.1f}g** / {limit_lemak_jenuh:.1f}g dari batas maksimal periode.")
            st.progress(min(int(percent_lemak_jenuh), 100))
            if percent_lemak_jenuh > 100:
                st.error(f"🔴 Peringatan! Konsumsi produk ini saja sudah **melebihi {percent_lemak_jenuh - 100:.0f}%** dari batas aman lemak jenuh Anda.")
            elif percent_lemak_jenuh > 50:
                st.warning(f"🟡 Perhatian. Konsumsi produk ini menggunakan **{percent_lemak_jenuh:.0f}%** dari alokasi lemak jenuh Anda untuk periode ini.")

            display_profile = kondisi_medis if kondisi_medis != "Tidak Ada" else f"{user_gender} {user_age} Thn"
            st.caption(f"Perhitungan berdasarkan profil '{display_profile}' (TDEE Dinamis) selama {simulation_period_months} bulan. Ingat, ini baru dari 1 produk, belum memperhitungkan asupan makanan berat Anda sehari-hari.")
