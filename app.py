from __future__ import annotations

import io
from datetime import datetime

import easyocr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from model_utils import (
    analyze_product_fully,
    detect_harmful_additives,
    load_prediction_models,
    preprocess_batch_excel_data,
)
from ocr_utils import parse_scan_result


st.set_page_config(
    page_title="SMART NutriScan AI",
    page_icon="🧠",
    layout="wide",
)


if "scan_history" not in st.session_state:
    st.session_state.scan_history = []


@st.cache_resource(show_spinner=False)
def load_all_models_and_scaler():
    return load_prediction_models()


@st.cache_resource(show_spinner=False)
def load_ocr_model():
    return easyocr.Reader(["id", "en"], gpu=False)


feat_model, lgbm_model, w2v_model, scaler = load_all_models_and_scaler()
reader = load_ocr_model()


NUTRITION_KEYS = [
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


def init_parsed_data():
    return {
        "product_name": "",
        "energi": 0.0,
        "lemak_total": 0.0,
        "lemak_jenuh": 0.0,
        "protein": 0.0,
        "karbohidrat": 0.0,
        "gula": 0.0,
        "garam": 0.0,
        "natrium": 0.0,
        "natrium_benzoat": 0.0,
        "komposisi": "",
    }


if "ocr_data" not in st.session_state:
    st.session_state.ocr_data = init_parsed_data()


def hitung_tdee_dinamis(gender, usia, berat, tinggi, aktivitas):
    if gender == "Pria":
        bmr = (10 * berat) + (6.25 * tinggi) - (5 * usia) + 5
    else:
        bmr = (10 * berat) + (6.25 * tinggi) - (5 * usia) - 161

    faktor = {
        "Sedentary": 1.2,
        "Ringan": 1.375,
        "Sedang": 1.55,
        "Aktif": 1.725,
        "Sangat Aktif": 1.9,
    }

    tdee = bmr * faktor.get(aktivitas, 1.2)
    return {
        "kalori": tdee,
        "gula": (tdee * 0.10) / 4,
        "lemak_jenuh": (tdee * 0.10) / 9,
        "natrium": 2000,
    }


def build_nutrition_data(
    energi,
    lemak_total,
    lemak_jenuh,
    protein,
    karbohidrat,
    gula,
    garam,
    natrium,
    natrium_benzoat,
):
    return {
        "energi": float(energi),
        "lemak_total": float(lemak_total),
        "lemak_jenuh": float(lemak_jenuh),
        "protein": float(protein),
        "karbohidrat": float(karbohidrat),
        "gula": float(gula),
        "garam": float(garam),
        "natrium": float(natrium),
        "natrium_benzoat": float(natrium_benzoat),
    }


def render_risk_status(risk_score):
    st.metric("Skor Risiko Prediksi", f"{risk_score:.2f}%")
    if risk_score >= 75:
        st.error("Risiko sangat tinggi")
    elif risk_score >= 50:
        st.warning("Risiko tinggi")
    elif risk_score >= 25:
        st.warning("Risiko sedang")
    else:
        st.success("Risiko rendah")


def render_xai_radar(xai_factors):
    categories = list(xai_factors.keys())
    if not categories:
        return

    norm_values = []
    for key, value in xai_factors.items():
        key_lower = key.lower()
        if "gula" in key_lower:
            norm_values.append(min((value / 50) * 100, 100))
        elif "natrium" in key_lower and "benzoat" not in key_lower:
            norm_values.append(min((value / 1500) * 100, 100))
        elif "lemak" in key_lower:
            norm_values.append(min((value / 67) * 100, 100))
        elif "energi" in key_lower:
            norm_values.append(min((value / 2000) * 100, 100))
        else:
            norm_values.append(min((value / 100) * 100, 100))

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=norm_values + [norm_values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Kandungan Produk",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
        showlegend=False,
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_health_metrics(nutrition_data, takaran_saji, current_threshold):
    st.markdown("### Profil Gizi Ringkas")

    energi = nutrition_data["energi"]
    gula = nutrition_data["gula"]
    natrium = nutrition_data["natrium"]
    lemak_jenuh = nutrition_data["lemak_jenuh"]

    kepadatan = energi / takaran_saji if takaran_saji > 0 else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Kepadatan Energi", f"{kepadatan:.2f} kkal/g")
    col2.metric("Gula per Saji", f"{gula:.1f} g")
    col3.metric("Natrium per Saji", f"{natrium:.0f} mg")

    st.write("Pemakaian batas harian berdasarkan profil pengguna:")
    gula_pct = (gula / current_threshold["gula"] * 100) if current_threshold["gula"] else 0
    natrium_pct = (natrium / current_threshold["natrium"] * 100) if current_threshold["natrium"] else 0
    lemak_jenuh_pct = (lemak_jenuh / current_threshold["lemak_jenuh"] * 100) if current_threshold["lemak_jenuh"] else 0

    st.write(f"Gula: {gula_pct:.1f}% dari batas harian")
    st.progress(min(int(gula_pct), 100))
    st.write(f"Natrium: {natrium_pct:.1f}% dari batas harian")
    st.progress(min(int(natrium_pct), 100))
    st.write(f"Lemak jenuh: {lemak_jenuh_pct:.1f}% dari batas harian")
    st.progress(min(int(lemak_jenuh_pct), 100))


def run_product_analysis(product_name, takaran_saji, nutrition_data, komposisi, current_threshold):
    risk_score, xai_factors, recommendation = analyze_product_fully(
        nutrition_data,
        komposisi,
        feat_model,
        lgbm_model,
        w2v_model,
        scaler,
    )

    st.session_state.scan_history.append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "product_name": product_name or "Produk Tanpa Nama",
        "risk_score": risk_score,
        "nutrition": nutrition_data,
    })

    render_risk_status(risk_score)
    st.markdown("#### Radar Kontribusi Nutrisi")
    render_xai_radar(xai_factors)
    st.markdown("#### Rekomendasi")
    st.info(recommendation)

    is_upf, flags = detect_harmful_additives(komposisi)
    if is_upf:
        st.error("Indikasi bahan ultra proses terdeteksi")
        st.write(", ".join(flags))

    st.markdown("---")
    render_health_metrics(nutrition_data, takaran_saji, current_threshold)


def input_form(prefix, defaults):
    product_name = st.text_input("Nama Produk", value=defaults.get("product_name", ""), key=f"{prefix}_name")

    c0, c1, c2 = st.columns(3)
    takaran_saji = c0.number_input("Takaran Saji g atau ml", min_value=1.0, value=30.0, format="%.1f", key=f"{prefix}_saji")
    energi = c1.number_input("Energi kkal", min_value=0.0, value=float(defaults.get("energi", 0)), format="%.1f", key=f"{prefix}_energi")
    lemak_total = c2.number_input("Lemak Total g", min_value=0.0, value=float(defaults.get("lemak_total", 0)), format="%.1f", key=f"{prefix}_lemak")

    c3, c4, c5 = st.columns(3)
    lemak_jenuh = c3.number_input("Lemak Jenuh g", min_value=0.0, value=float(defaults.get("lemak_jenuh", 0)), format="%.1f", key=f"{prefix}_jenuh")
    protein = c4.number_input("Protein g", min_value=0.0, value=float(defaults.get("protein", 0)), format="%.1f", key=f"{prefix}_protein")
    karbohidrat = c5.number_input("Karbohidrat g", min_value=0.0, value=float(defaults.get("karbohidrat", 0)), format="%.1f", key=f"{prefix}_karbo")

    c6, c7, c8, c9 = st.columns(4)
    gula = c6.number_input("Gula g", min_value=0.0, value=float(defaults.get("gula", 0)), format="%.1f", key=f"{prefix}_gula")
    garam = c7.number_input("Garam g", min_value=0.0, value=float(defaults.get("garam", 0)), format="%.2f", key=f"{prefix}_garam")
    natrium = c8.number_input("Natrium mg", min_value=0.0, value=float(defaults.get("natrium", 0)), format="%.0f", key=f"{prefix}_natrium")
    natrium_benzoat = c9.number_input("Natrium Benzoat mg", min_value=0.0, value=float(defaults.get("natrium_benzoat", 0)), format="%.2f", key=f"{prefix}_benzoat")

    komposisi = st.text_area("Komposisi", value=defaults.get("komposisi", ""), height=120, key=f"{prefix}_komposisi")

    nutrition_data = build_nutrition_data(
        energi,
        lemak_total,
        lemak_jenuh,
        protein,
        karbohidrat,
        gula,
        garam,
        natrium,
        natrium_benzoat,
    )

    return product_name, takaran_saji, nutrition_data, komposisi


with st.sidebar:
    try:
        st.image("assets/Logo Smart NutriScan AI.png", width=150)
    except Exception:
        st.markdown("## SMART NutriScan AI")

    st.title("SMART NutriScan AI")
    st.header("Profil Pengguna")

    col_gender, col_age = st.columns(2)
    user_gender = col_gender.selectbox("Gender", ["Pria", "Wanita"])
    user_age = col_age.number_input("Usia", min_value=1, max_value=120, value=25)

    col_weight, col_height = st.columns(2)
    user_weight = col_weight.number_input("Berat kg", min_value=10.0, max_value=300.0, value=65.0)
    user_height = col_height.number_input("Tinggi cm", min_value=50.0, max_value=250.0, value=165.0)

    user_activity = st.selectbox("Aktivitas", ["Sedentary", "Ringan", "Sedang", "Aktif", "Sangat Aktif"])
    kondisi_medis = st.selectbox("Kondisi Khusus", ["Tidak Ada", "Penderita Hipertensi", "Risiko Penyakit Ginjal", "Anak anak"])

    current_threshold = hitung_tdee_dinamis(user_gender, user_age, user_weight, user_height, user_activity)
    if kondisi_medis == "Penderita Hipertensi":
        current_threshold["natrium"] = 1200
    elif kondisi_medis == "Risiko Penyakit Ginjal":
        current_threshold["natrium"] = 1000
        current_threshold["kalori"] *= 0.9
    elif kondisi_medis == "Anak anak":
        current_threshold["gula"] = 25
        current_threshold["natrium"] = 1500

    with st.expander("Lihat batas harian"):
        st.write(f"Kalori: {current_threshold['kalori']:.0f} kkal")
        st.write(f"Gula: {current_threshold['gula']:.1f} g")
        st.write(f"Lemak jenuh: {current_threshold['lemak_jenuh']:.1f} g")
        st.write(f"Natrium: {current_threshold['natrium']} mg")

    app_mode = st.radio(
        "Pilih Fitur",
        [
            "Analisis Produk Tunggal",
            "Scan from Image",
            "Analisis Batch Excel",
            "Riwayat Analisis",
            "Edukasi Gizi",
        ],
    )


st.title("SMART NutriScan AI")
st.caption("Analisis produk pangan berbasis OCR, machine learning, dan konfirmasi data manual.")

model_ready = all([feat_model, lgbm_model, w2v_model, scaler])
if model_ready:
    st.success("Model utama berhasil dimuat.")
else:
    st.warning("Sebagian model utama belum terbaca. Aplikasi tetap berjalan dengan analisis cadangan jika dibutuhkan.")


if app_mode == "Analisis Produk Tunggal":
    st.header("Analisis Produk Tunggal")
    defaults = {
        "product_name": "Biskuit Cokelat",
        "energi": 180.0,
        "lemak_total": 8.0,
        "lemak_jenuh": 4.0,
        "protein": 2.0,
        "karbohidrat": 25.0,
        "gula": 15.0,
        "garam": 0.3,
        "natrium": 200.0,
        "natrium_benzoat": 0.0,
        "komposisi": "Tepung terigu, gula, minyak nabati, cokelat bubuk, pengembang, perisa sintetik, garam.",
    }

    product_name, takaran_saji, nutrition_data, komposisi = input_form("manual", defaults)

    if st.button("Analisis AI dan Gizi", type="primary"):
        run_product_analysis(product_name, takaran_saji, nutrition_data, komposisi, current_threshold)


elif app_mode == "Scan from Image":
    st.header("Scan Produk Otomatis")
    st.info("Ambil foto dekat, lurus, tidak blur, dan pastikan label memenuhi sebagian besar area gambar. Setelah OCR selesai, koreksi data sebelum analisis.")

    col_scan1, col_scan2 = st.columns(2)

    with col_scan1:
        st.subheader("Scan 1: Informasi Nilai Gizi")
        input_type_1 = st.radio("Metode input nilai gizi", ["Upload File", "Kamera Langsung"], key="input_gizi")
        img_file_1 = st.file_uploader("Upload foto nilai gizi", type=["jpg", "jpeg", "png"], key="upload_gizi") if input_type_1 == "Upload File" else st.camera_input("Foto nilai gizi", key="camera_gizi")

        if img_file_1 is not None:
            image_1 = Image.open(img_file_1)
            st.image(image_1, caption="Gambar nilai gizi", use_container_width=True)

            with st.spinner("Membaca nilai gizi dengan OCR multi preprocessing..."):
                scan_result_1 = parse_scan_result(reader, image_1, mode="nutrition")
                parsed_gizi = scan_result_1["parsed"]

                for key, value in parsed_gizi.items():
                    if key in st.session_state.ocr_data:
                        if value not in [0, 0.0, "Tidak terdeteksi.", "Produk Tanpa Nama", ""]:
                            st.session_state.ocr_data[key] = value

            st.success("Nilai gizi berhasil diproses. Periksa lagi hasilnya di form konfirmasi.")
            with st.expander("Lihat teks OCR nilai gizi"):
                st.text(scan_result_1["raw_text"] or "Tidak ada teks terbaca")
            with st.expander("Lihat variasi preprocessing"):
                for name, img in scan_result_1["variants"].items():
                    st.image(img, caption=name, use_container_width=True)

    with col_scan2:
        st.subheader("Scan 2: Komposisi Produk")
        input_type_2 = st.radio("Metode input komposisi", ["Upload File", "Kamera Langsung"], key="input_komposisi")
        img_file_2 = st.file_uploader("Upload foto komposisi", type=["jpg", "jpeg", "png"], key="upload_komposisi") if input_type_2 == "Upload File" else st.camera_input("Foto komposisi", key="camera_komposisi")

        if img_file_2 is not None:
            image_2 = Image.open(img_file_2)
            st.image(image_2, caption="Gambar komposisi", use_container_width=True)

            with st.spinner("Membaca komposisi dengan OCR multi preprocessing..."):
                scan_result_2 = parse_scan_result(reader, image_2, mode="composition")
                parsed_komposisi = scan_result_2["parsed"].get("komposisi", "Tidak terdeteksi.")
                if parsed_komposisi != "Tidak terdeteksi.":
                    st.session_state.ocr_data["komposisi"] = parsed_komposisi

            st.success("Komposisi berhasil diproses. Periksa lagi hasilnya di form konfirmasi.")
            with st.expander("Lihat teks OCR komposisi"):
                st.text(scan_result_2["raw_text"] or "Tidak ada teks terbaca")
            with st.expander("Lihat variasi preprocessing"):
                for name, img in scan_result_2["variants"].items():
                    st.image(img, caption=name, use_container_width=True)

    st.markdown("---")
    st.subheader("Konfirmasi Data Hasil OCR")
    st.warning("Jangan langsung percaya OCR mentah. Koreksi angka dan komposisi sebelum menjalankan rekomendasi.")

    product_name, takaran_saji, nutrition_data, komposisi = input_form("ocr", st.session_state.ocr_data)

    if st.button("Analisis dari Data Hasil OCR", type="primary"):
        run_product_analysis(product_name, takaran_saji, nutrition_data, komposisi, current_threshold)


elif app_mode == "Analisis Batch Excel":
    st.header("Analisis Batch Excel")
    st.write("Upload file Excel dengan kolom Nama Produk, Energi, Lemak, Karbohidrat, Gula, Protein, Garam, Natrium, Natrium Benzoat, dan Komposisi jika tersedia.")

    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        df_clean = preprocess_batch_excel_data(df)
        results = []

        for _, row in df_clean.iterrows():
            nutrition_data = {
                "energi": row.get("Energi", 0),
                "lemak_total": row.get("Lemak", 0),
                "lemak_jenuh": row.get("Lemak Jenuh", 0),
                "protein": row.get("Protein", 0),
                "karbohidrat": row.get("Karbohidrat", 0),
                "gula": row.get("Gula", 0),
                "garam": row.get("Garam", 0),
                "natrium": row.get("Natrium", 0),
                "natrium_benzoat": row.get("Natrium Benzoat", 0),
            }
            komposisi = row.get("Komposisi", "")
            risk_score, _, recommendation = analyze_product_fully(
                nutrition_data,
                komposisi,
                feat_model,
                lgbm_model,
                w2v_model,
                scaler,
            )
            results.append({
                "Nama Produk": row.get("Nama Produk", row.get("Produk", "Produk Tanpa Nama")),
                "Skor Risiko": round(risk_score, 2),
                "Rekomendasi": recommendation,
            })

        result_df = pd.DataFrame(results)
        st.dataframe(result_df, use_container_width=True)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            result_df.to_excel(writer, index=False, sheet_name="Hasil Analisis")
        st.download_button("Download Hasil Excel", output.getvalue(), "hasil_analisis_nutriscan.xlsx")


elif app_mode == "Riwayat Analisis":
    st.header("Riwayat Analisis")
    if st.session_state.scan_history:
        st.dataframe(pd.DataFrame(st.session_state.scan_history), use_container_width=True)
    else:
        st.info("Belum ada riwayat analisis pada sesi ini.")


elif app_mode == "Edukasi Gizi":
    st.header("Edukasi Gizi")
    st.markdown(
        """
        **Cara membaca hasil aplikasi:**

        1. OCR hanya membantu mengisi data awal, bukan pengganti validasi pengguna.
        2. Gula tinggi perlu diperhatikan karena berpengaruh pada beban asupan harian.
        3. Natrium tinggi perlu dibatasi, terutama pada pengguna dengan risiko hipertensi.
        4. Lemak jenuh tinggi sebaiknya tidak dikonsumsi terlalu sering.
        5. Komposisi dengan pemanis buatan, pewarna sintetik, pengawet, dan penguat rasa menandakan indikasi produk ultra proses.
        """
    )
