from __future__ import annotations

import io
from datetime import datetime

import easyocr
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from model_utils import (
    analyze_product_fully,
    classify_risk,
    detect_harmful_additives,
    has_sufficient_input,
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

# EasyOCR tidak dimuat saat aplikasi pertama dibuka.
# Model OCR baru dimuat ketika tombol proses OCR ditekan agar upload gambar tidak langsung membuat app crash.
def get_ocr_reader_safely():
    try:
        return load_ocr_model(), None
    except Exception as exc:
        return None, str(exc)


def safe_image(image, caption=None):
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image, caption=caption, use_column_width=True)


def fmt(value, digits=2, suffix=""):
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return f"0.{'0' * digits}{suffix}"


def run_ocr_safely(reader, image, mode):
    try:
        return parse_scan_result(reader, image, mode=mode), None
    except Exception as exc:
        return None, str(exc)


OCR_WIDGET_KEY_MAP = {
    "takaran_saji": "ocr_saji",
    "energi": "ocr_energi",
    "lemak_total": "ocr_lemak",
    "lemak_jenuh": "ocr_jenuh",
    "protein": "ocr_protein",
    "karbohidrat": "ocr_karbo",
    "gula": "ocr_gula",
    "garam": "ocr_garam",
    "natrium": "ocr_natrium",
    "natrium_benzoat": "ocr_benzoat",
    "komposisi": "ocr_komposisi",
    "product_name": "ocr_name",
}


LEGACY_OCR_WIDGET_KEYS = set(OCR_WIDGET_KEY_MAP.values())

# Membersihkan key lama dari versi sebelumnya.
# Ini mencegah warning Streamlit: widget dibuat dengan default value, tetapi juga diisi lewat Session State.
for _legacy_key in list(LEGACY_OCR_WIDGET_KEYS):
    if _legacy_key in st.session_state:
        del st.session_state[_legacy_key]


def set_widget_default(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def sync_ocr_value_to_form(key, value):
    """Menyinkronkan hasil OCR hanya ke data sumber.

    Catatan teknis: jangan menulis langsung ke key widget number_input atau text_area
    setelah widget pernah dibuat, karena Streamlit dapat memunculkan error runtime.
    Form dibuat ulang memakai versi key yang berubah setelah OCR berhasil.
    """
    if key not in st.session_state.ocr_data:
        return

    st.session_state.ocr_data[key] = value


def bump_ocr_form_version():
    st.session_state.ocr_form_version += 1


def clear_ocr_analysis_result():
    st.session_state.ocr_analysis_result = None


def render_ocr_result_debug(scan_result, label):
    if not scan_result:
        return

    errors = scan_result.get("errors", [])
    quality_warnings = scan_result.get("quality_warnings", [])

    if quality_warnings:
        with st.expander(f"Catatan kualitas OCR {label}", expanded=False):
            for item in quality_warnings:
                st.warning(item)

    if errors:
        with st.expander(f"Catatan error OCR {label}", expanded=False):
            for item in errors:
                st.warning(item)

    with st.expander(f"Lihat teks OCR {label}", expanded=False):
        st.caption(f"Variasi gambar terbaik: {scan_result.get('best_variant', 'tidak diketahui')}")
        st.text(scan_result.get("raw_text") or "Tidak ada teks terbaca")

    with st.expander(f"Ringkasan variasi preprocessing {label}", expanded=False):
        names = list(scan_result.get("variants", {}).keys())
        st.write(names if names else "Tidak ada variasi gambar tersimpan.")


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


EXAMPLE_PRESETS = {
    "Kosong": {
        "product_name": "",
        "takaran_saji": 100.0,
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
    },
    "Contoh Aman": {
        "product_name": "Contoh Produk Aman",
        "takaran_saji": 100.0,
        "energi": 180.0,
        "lemak_total": 5.0,
        "lemak_jenuh": 1.5,
        "protein": 8.0,
        "karbohidrat": 22.0,
        "gula": 6.0,
        "garam": 0.5,
        "natrium": 150.0,
        "natrium_benzoat": 10.0,
        "komposisi": "Tepung, susu, gula, garam.",
    },
    "Contoh Sedang": {
        "product_name": "Contoh Produk Sedang",
        "takaran_saji": 100.0,
        "energi": 320.0,
        "lemak_total": 15.0,
        "lemak_jenuh": 6.0,
        "protein": 13.0,
        "karbohidrat": 45.0,
        "gula": 22.0,
        "garam": 1.8,
        "natrium": 600.0,
        "natrium_benzoat": 130.0,
        "komposisi": "Tepung terigu, gula, minyak nabati, cokelat bubuk, pengembang, garam.",
    },
    "Contoh Tinggi": {
        "product_name": "Contoh Produk Tinggi",
        "takaran_saji": 100.0,
        "energi": 550.0,
        "lemak_total": 30.0,
        "lemak_jenuh": 14.0,
        "protein": 22.0,
        "karbohidrat": 75.0,
        "gula": 45.0,
        "garam": 4.0,
        "natrium": 1500.0,
        "natrium_benzoat": 300.0,
        "komposisi": "Gula, sirup fruktosa, minyak nabati terhidrogenasi, penguat rasa, pengawet natrium benzoat, perisa sintetik.",
    },
}


def init_parsed_data():
    return dict(EXAMPLE_PRESETS["Kosong"])


if "ocr_data" not in st.session_state:
    st.session_state.ocr_data = init_parsed_data()

if "ocr_form_version" not in st.session_state:
    st.session_state.ocr_form_version = 0

if "ocr_analysis_result" not in st.session_state:
    st.session_state.ocr_analysis_result = None

if "manual_analysis_result" not in st.session_state:
    st.session_state.manual_analysis_result = None


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
    risk_info = classify_risk(risk_score)
    st.metric("Skor Risiko Prediksi", f"{risk_score:.2f}%")
    st.metric("Klasifikasi", risk_info["label"])

    if risk_info["style"] == "success":
        st.success(f"Klasifikasi {risk_info['label']}")
    elif risk_info["style"] == "warning":
        st.warning(f"Klasifikasi {risk_info['label']}")
    else:
        st.error(f"Klasifikasi {risk_info['label']}")


def render_xai_radar(xai_factors):
    categories = list(xai_factors.keys())
    if not categories:
        return

    norm_values = []
    for key, value in xai_factors.items():
        key_lower = key.lower()
        value = float(value or 0)
        if "gula" in key_lower:
            norm_values.append(min((value / 45) * 100, 100))
        elif "natrium" in key_lower and "benzoat" not in key_lower:
            norm_values.append(min((value / 1500) * 100, 100))
        elif "benzoat" in key_lower:
            norm_values.append(min((value / 300) * 100, 100))
        elif "lemak total" in key_lower:
            norm_values.append(min((value / 30) * 100, 100))
        elif "lemak jenuh" in key_lower:
            norm_values.append(min((value / 14) * 100, 100))
        elif "energi" in key_lower:
            norm_values.append(min((value / 550) * 100, 100))
        elif "karbohidrat" in key_lower:
            norm_values.append(min((value / 75) * 100, 100))
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
    st.markdown("### Profil Gizi dan Makronutrien")

    energi = float(nutrition_data["energi"])
    gula = float(nutrition_data["gula"])
    natrium = float(nutrition_data["natrium"])
    lemak_jenuh = float(nutrition_data["lemak_jenuh"])
    karbohidrat = float(nutrition_data["karbohidrat"])

    kepadatan = energi / takaran_saji if takaran_saji > 0 else 0
    rasio_gula_karbo = (gula / karbohidrat * 100) if karbohidrat > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Kepadatan Energi", f"{kepadatan:.2f} kkal/g")
    col2.metric("Rasio Gula dari Karbohidrat", f"{rasio_gula_karbo:.2f}%")
    col3.metric("Natrium per Saji", f"{natrium:.0f} mg")

    st.write("Pemenuhan angka kecukupan gizi harian berdasarkan profil pengguna:")
    gula_pct = (gula / current_threshold["gula"] * 100) if current_threshold["gula"] else 0
    natrium_pct = (natrium / current_threshold["natrium"] * 100) if current_threshold["natrium"] else 0
    lemak_jenuh_pct = (lemak_jenuh / current_threshold["lemak_jenuh"] * 100) if current_threshold["lemak_jenuh"] else 0

    st.write(f"Gula: {gula:.2f} g dari batas {current_threshold['gula']:.2f} g per hari. Persentase: {gula_pct:.2f}%")
    st.progress(min(int(round(gula_pct)), 100))
    st.write(f"Natrium: {natrium:.0f} mg dari batas {current_threshold['natrium']:.0f} mg per hari. Persentase: {natrium_pct:.2f}%")
    st.progress(min(int(round(natrium_pct)), 100))
    st.write(f"Lemak jenuh: {lemak_jenuh:.2f} g dari batas {current_threshold['lemak_jenuh']:.2f} g per hari. Persentase: {lemak_jenuh_pct:.2f}%")
    st.progress(min(int(round(lemak_jenuh_pct)), 100))


def make_analysis_signature(product_name, takaran_saji, nutrition_data, komposisi):
    return {
        "product_name": str(product_name or "").strip(),
        "takaran_saji": round(float(takaran_saji or 0), 4),
        "nutrition": {
            key: round(float(nutrition_data.get(key, 0) or 0), 4)
            for key in sorted(nutrition_data.keys())
        },
        "komposisi": str(komposisi or "").strip(),
    }


def build_analysis_result(product_name, takaran_saji, nutrition_data, komposisi):
    if not has_sufficient_input(nutrition_data):
        return {
            "status": "insufficient",
            "message": "Data belum cukup untuk dianalisis. Isi atau koreksi minimal satu nilai gizi yang valid sebelum menjalankan rekomendasi.",
            "integrity_note": "Sistem tidak memberi label tinggi, sedang, atau aman ketika data masih kosong. Ini menjaga integritas hasil analisis.",
            "product_name": product_name or "Produk Tanpa Nama",
            "takaran_saji": float(takaran_saji or 0),
            "nutrition_data": nutrition_data,
            "komposisi": komposisi,
        }

    risk_score, xai_factors, recommendation = analyze_product_fully(
        nutrition_data,
        komposisi,
        feat_model,
        lgbm_model,
        w2v_model,
        scaler,
    )
    risk_info = classify_risk(risk_score)
    is_upf, flags = detect_harmful_additives(komposisi)

    return {
        "status": "ok",
        "product_name": product_name or "Produk Tanpa Nama",
        "takaran_saji": float(takaran_saji or 0),
        "nutrition_data": nutrition_data,
        "komposisi": komposisi,
        "risk_score": float(risk_score),
        "risk_info": risk_info,
        "xai_factors": xai_factors,
        "recommendation": recommendation,
        "is_upf": is_upf,
        "upf_flags": flags,
    }


def render_analysis_result(analysis_result, current_threshold, current_signature=None):
    if not analysis_result:
        st.info("Hasil analisis akan muncul di sini setelah tombol analisis diklik.")
        return

    stored_signature = analysis_result.get("input_signature")
    if stored_signature is not None and current_signature is not None and stored_signature != current_signature:
        st.warning("Data input sudah berubah setelah analisis terakhir. Klik tombol analisis lagi untuk memperbarui hasil.")

    if analysis_result.get("status") == "insufficient":
        st.warning(analysis_result.get("message", "Data belum cukup untuk dianalisis."))
        st.info(analysis_result.get("integrity_note", "Periksa kembali data input sebelum analisis."))
        return

    risk_score = float(analysis_result.get("risk_score", 0))
    xai_factors = analysis_result.get("xai_factors", {})
    recommendation = analysis_result.get("recommendation", "")
    nutrition_data = analysis_result.get("nutrition_data", {})
    takaran_saji = analysis_result.get("takaran_saji", 0)
    komposisi = analysis_result.get("komposisi", "")

    render_risk_status(risk_score)
    st.markdown("#### Radar Kontribusi Nutrisi")
    render_xai_radar(xai_factors)
    st.markdown("#### Rekomendasi")
    st.info(recommendation)

    if analysis_result.get("is_upf"):
        st.error("Indikasi bahan ultra proses terdeteksi")
        st.write(", ".join(analysis_result.get("upf_flags", [])))

    st.markdown("---")
    render_health_metrics(nutrition_data, takaran_saji, current_threshold)


def store_product_analysis_result(product_name, takaran_saji, nutrition_data, komposisi, store_key, input_signature=None):
    """Menganalisis produk lalu menyimpan hasil tanpa langsung merender output.

    Streamlit selalu melakukan rerun setelah tombol diklik. Jika hasil langsung dirender
    di bawah tombol dan juga dirender di panel hasil, tampilan menjadi dobel. Fungsi ini
    menjaga satu sumber tampilan hasil, yaitu panel Hasil Analisis AI di samping form.
    """
    analysis_result = build_analysis_result(product_name, takaran_saji, nutrition_data, komposisi)
    analysis_result["input_signature"] = input_signature or make_analysis_signature(product_name, takaran_saji, nutrition_data, komposisi)

    st.session_state[store_key] = analysis_result

    if analysis_result.get("status") == "ok":
        risk_score = analysis_result["risk_score"]
        risk_info = analysis_result["risk_info"]
        st.session_state.scan_history.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "product_name": analysis_result["product_name"],
            "risk_score": round(risk_score, 2),
            "classification": risk_info["label"],
            "nutrition": nutrition_data,
        })

    return analysis_result


def run_product_analysis(product_name, takaran_saji, nutrition_data, komposisi, current_threshold, store_key=None, input_signature=None):
    """Fungsi lama tetap disediakan untuk kompatibilitas internal.

    Untuk fitur utama, gunakan store_product_analysis_result agar hasil tidak tampil dobel.
    """
    target_key = store_key or "manual_analysis_result"
    analysis_result = store_product_analysis_result(
        product_name,
        takaran_saji,
        nutrition_data,
        komposisi,
        target_key,
        input_signature=input_signature,
    )
    render_analysis_result(analysis_result, current_threshold, current_signature=analysis_result["input_signature"])
    return analysis_result

def input_form(prefix, defaults):
    """Form input yang aman terhadap Session State Streamlit.

    Prinsipnya: nilai default dimasukkan ke session_state sebelum widget dibuat,
    lalu widget tidak lagi diberi parameter value. Cara ini menghapus warning
    default value versus Session State API.
    """
    name_key = f"{prefix}_name"
    saji_key = f"{prefix}_saji"
    energi_key = f"{prefix}_energi"
    lemak_key = f"{prefix}_lemak"
    jenuh_key = f"{prefix}_jenuh"
    protein_key = f"{prefix}_protein"
    karbo_key = f"{prefix}_karbo"
    gula_key = f"{prefix}_gula"
    garam_key = f"{prefix}_garam"
    natrium_key = f"{prefix}_natrium"
    benzoat_key = f"{prefix}_benzoat"
    komposisi_key = f"{prefix}_komposisi"

    set_widget_default(name_key, str(defaults.get("product_name", "")))
    set_widget_default(saji_key, float(defaults.get("takaran_saji", 100.0)))
    set_widget_default(energi_key, float(defaults.get("energi", 0.0)))
    set_widget_default(lemak_key, float(defaults.get("lemak_total", 0.0)))
    set_widget_default(jenuh_key, float(defaults.get("lemak_jenuh", 0.0)))
    set_widget_default(protein_key, float(defaults.get("protein", 0.0)))
    set_widget_default(karbo_key, float(defaults.get("karbohidrat", 0.0)))
    set_widget_default(gula_key, float(defaults.get("gula", 0.0)))
    set_widget_default(garam_key, float(defaults.get("garam", 0.0)))
    set_widget_default(natrium_key, float(defaults.get("natrium", 0.0)))
    set_widget_default(benzoat_key, float(defaults.get("natrium_benzoat", 0.0)))
    set_widget_default(komposisi_key, str(defaults.get("komposisi", "")))

    product_name = st.text_input("Nama Produk", key=name_key)

    c0, c1, c2 = st.columns(3)
    takaran_saji = c0.number_input("Takaran Saji g atau ml", min_value=1.0, format="%.2f", key=saji_key)
    energi = c1.number_input("Energi kkal", min_value=0.0, format="%.2f", key=energi_key)
    lemak_total = c2.number_input("Lemak Total g", min_value=0.0, format="%.2f", key=lemak_key)

    c3, c4, c5 = st.columns(3)
    lemak_jenuh = c3.number_input("Lemak Jenuh g", min_value=0.0, format="%.2f", key=jenuh_key)
    protein = c4.number_input("Protein g", min_value=0.0, format="%.2f", key=protein_key)
    karbohidrat = c5.number_input("Karbohidrat g", min_value=0.0, format="%.2f", key=karbo_key)

    c6, c7, c8, c9 = st.columns(4)
    gula = c6.number_input("Gula g", min_value=0.0, format="%.2f", key=gula_key)
    garam = c7.number_input("Garam g", min_value=0.0, format="%.2f", key=garam_key)
    natrium = c8.number_input("Natrium mg", min_value=0.0, format="%.2f", key=natrium_key)
    natrium_benzoat = c9.number_input("Natrium Benzoat mg", min_value=0.0, format="%.2f", key=benzoat_key)

    komposisi = st.text_area("Komposisi", height=120, key=komposisi_key)

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
        st.write(f"Kalori: {current_threshold['kalori']:.2f} kkal")
        st.write(f"Gula: {current_threshold['gula']:.2f} g")
        st.write(f"Lemak jenuh: {current_threshold['lemak_jenuh']:.2f} g")
        st.write(f"Natrium: {current_threshold['natrium']:.2f} mg")

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
st.caption("Analisis produk pangan berbasis OCR, machine learning, aturan gizi terkalibrasi, dan konfirmasi data manual.")

model_ready = all([feat_model, lgbm_model, w2v_model, scaler])
if model_ready:
    st.success("Model utama berhasil dimuat. Skor tetap dijaga oleh aturan gizi agar klasifikasi konsisten.")
else:
    st.warning("Sebagian model utama belum terbaca. Aplikasi tetap berjalan dengan analisis gizi terkalibrasi.")


if app_mode == "Analisis Produk Tunggal":
    st.header("Analisis Produk Tunggal")

    manual_input_col, manual_result_col = st.columns([1.15, 1], gap="large")

    with manual_input_col:
        st.subheader("Input Informasi Produk")
        preset_name = st.selectbox("Pilih contoh uji atau isi manual", list(EXAMPLE_PRESETS.keys()), index=1)
        defaults = dict(EXAMPLE_PRESETS[preset_name])

        product_name, takaran_saji, nutrition_data, komposisi = input_form("manual", defaults)
        manual_signature = make_analysis_signature(product_name, takaran_saji, nutrition_data, komposisi)

        if st.button("Analisis AI dan Gizi", type="primary"):
            store_product_analysis_result(
                product_name,
                takaran_saji,
                nutrition_data,
                komposisi,
                store_key="manual_analysis_result",
                input_signature=manual_signature,
            )
            st.success("Analisis berhasil diperbarui. Hasil ditampilkan di panel kanan.")

    with manual_result_col:
        st.subheader("Hasil Analisis AI (Prediksi Risiko)")
        render_analysis_result(st.session_state.manual_analysis_result, current_threshold, current_signature=manual_signature)


elif app_mode == "Scan from Image":
    st.header("Scan Produk Otomatis")
    st.info("Ambil foto dekat, lurus, tidak blur, dan pastikan label memenuhi sebagian besar area gambar. Setelah OCR selesai, koreksi data sebelum analisis.")

    if st.button("Reset Hasil OCR"):
        st.session_state.ocr_data = init_parsed_data()
        clear_ocr_analysis_result()
        bump_ocr_form_version()
        st.success("Hasil OCR dan analisis terakhir sudah dikosongkan.")

    col_scan1, col_scan2 = st.columns(2)

    with col_scan1:
        st.subheader("Scan 1: Informasi Nilai Gizi")
        input_type_1 = st.radio("Metode input nilai gizi", ["Upload File", "Kamera Langsung"], key="input_gizi")
        img_file_1 = st.file_uploader("Upload foto nilai gizi", type=["jpg", "jpeg", "png"], key="upload_gizi") if input_type_1 == "Upload File" else st.camera_input("Foto nilai gizi", key="camera_gizi")

        if img_file_1 is not None:
            try:
                image_1 = Image.open(img_file_1)
                safe_image(image_1, caption="Gambar nilai gizi")

                if st.button("Proses OCR Nilai Gizi", key="btn_ocr_gizi"):
                    with st.spinner("Mempersiapkan OCR dan membaca nilai gizi secara bertahap..."):
                        reader, reader_error = get_ocr_reader_safely()
                        if reader_error:
                            scan_result_1, ocr_error_1 = None, reader_error
                        else:
                            scan_result_1, ocr_error_1 = run_ocr_safely(reader, image_1, mode="nutrition")

                    if ocr_error_1:
                        st.error("OCR nilai gizi gagal diproses. Aplikasi tidak dihentikan. Silakan input manual atau coba foto yang lebih jelas.")
                        with st.expander("Detail error OCR nilai gizi"):
                            st.code(ocr_error_1)
                    else:
                        parsed_gizi = scan_result_1["parsed"]
                        changed = False
                        for key, value in parsed_gizi.items():
                            if key in st.session_state.ocr_data:
                                if value not in [0, 0.0, "Tidak terdeteksi.", "Produk Tanpa Nama", ""]:
                                    sync_ocr_value_to_form(key, value)
                                    changed = True
                        if changed:
                            clear_ocr_analysis_result()
                            bump_ocr_form_version()

                        st.success("Nilai gizi berhasil diproses. Angka satuan g yang terbaca sebagai 9 sudah dikoreksi sebelum masuk form. Periksa lagi sebelum analisis.")
                        render_ocr_result_debug(scan_result_1, "nilai gizi")
            except Exception as exc:
                st.error("Gambar nilai gizi tidak bisa dibaca. Coba upload ulang dalam format JPG atau PNG.")
                with st.expander("Detail error gambar nilai gizi"):
                    st.code(str(exc))

    with col_scan2:
        st.subheader("Scan 2: Komposisi Produk")
        input_type_2 = st.radio("Metode input komposisi", ["Upload File", "Kamera Langsung"], key="input_komposisi")
        img_file_2 = st.file_uploader("Upload foto komposisi", type=["jpg", "jpeg", "png"], key="upload_komposisi") if input_type_2 == "Upload File" else st.camera_input("Foto komposisi", key="camera_komposisi")

        if img_file_2 is not None:
            try:
                image_2 = Image.open(img_file_2)
                safe_image(image_2, caption="Gambar komposisi")

                if st.button("Proses OCR Komposisi", key="btn_ocr_komposisi"):
                    with st.spinner("Mempersiapkan OCR dan membaca komposisi secara bertahap..."):
                        reader, reader_error = get_ocr_reader_safely()
                        if reader_error:
                            scan_result_2, ocr_error_2 = None, reader_error
                        else:
                            scan_result_2, ocr_error_2 = run_ocr_safely(reader, image_2, mode="composition")

                    if ocr_error_2:
                        st.error("OCR komposisi gagal diproses. Aplikasi tidak dihentikan. Silakan input manual atau coba foto yang lebih jelas.")
                        with st.expander("Detail error OCR komposisi"):
                            st.code(ocr_error_2)
                    else:
                        parsed_komposisi = scan_result_2["parsed"].get("komposisi", "Tidak terdeteksi.")
                        if parsed_komposisi != "Tidak terdeteksi.":
                            sync_ocr_value_to_form("komposisi", parsed_komposisi)
                            clear_ocr_analysis_result()
                            bump_ocr_form_version()

                        st.success("Komposisi berhasil diproses dari satu variasi OCR terbaik agar tidak berulang. Periksa lagi sebelum analisis.")
                        render_ocr_result_debug(scan_result_2, "komposisi")
            except Exception as exc:
                st.error("Gambar komposisi tidak bisa dibaca. Coba upload ulang dalam format JPG atau PNG.")
                with st.expander("Detail error gambar komposisi"):
                    st.code(str(exc))

    st.markdown("---")
    input_col, result_col = st.columns([1.15, 1], gap="large")

    with input_col:
        st.subheader("Konfirmasi Data Input (Hasil OCR)")
        st.warning("Jangan langsung percaya OCR mentah. Koreksi angka dan komposisi sebelum menjalankan rekomendasi.")

        ocr_prefix = f"ocr_{st.session_state.ocr_form_version}"
        product_name, takaran_saji, nutrition_data, komposisi = input_form(ocr_prefix, st.session_state.ocr_data)
        ocr_signature = make_analysis_signature(product_name, takaran_saji, nutrition_data, komposisi)

        if st.button("Analisis dari Data Hasil OCR", type="primary"):
            store_product_analysis_result(
                product_name,
                takaran_saji,
                nutrition_data,
                komposisi,
                store_key="ocr_analysis_result",
                input_signature=ocr_signature,
            )
            st.success("Analisis berhasil diperbarui. Hasil ditampilkan di panel kanan.")

    with result_col:
        st.subheader("Hasil Analisis AI (Prediksi Risiko)")
        render_analysis_result(st.session_state.ocr_analysis_result, current_threshold, current_signature=ocr_signature)


elif app_mode == "Analisis Batch Excel":
    st.header("Analisis Batch Excel")
    st.write("Upload file Excel dengan kolom Nama Produk, Energi, Lemak, Lemak Jenuh, Karbohidrat, Gula, Protein, Garam, Natrium, Natrium Benzoat, dan Komposisi jika tersedia.")

    uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        df_clean = preprocess_batch_excel_data(df)
        results = []

        for _, row in df_clean.iterrows():
            nutrition_data = {
                "energi": row.get("Energi", 0),
                "lemak_total": row.get("Lemak", row.get("Lemak Total", 0)),
                "lemak_jenuh": row.get("Lemak Jenuh", 0),
                "protein": row.get("Protein", 0),
                "karbohidrat": row.get("Karbohidrat", 0),
                "gula": row.get("Gula", 0),
                "garam": row.get("Garam", 0),
                "natrium": row.get("Natrium", 0),
                "natrium_benzoat": row.get("Natrium Benzoat", 0),
            }
            komposisi = row.get("Komposisi", "")
            product_name = row.get("Nama Produk", row.get("Produk", "Produk Tanpa Nama"))

            if not has_sufficient_input(nutrition_data):
                results.append({
                    "Nama Produk": product_name,
                    "Skor Risiko": "Data belum cukup",
                    "Klasifikasi": "Belum dianalisis",
                    "Rekomendasi": "Isi nilai gizi yang valid sebelum analisis.",
                })
                continue

            risk_score, _, recommendation = analyze_product_fully(
                nutrition_data,
                komposisi,
                feat_model,
                lgbm_model,
                w2v_model,
                scaler,
            )
            risk_info = classify_risk(risk_score)
            results.append({
                "Nama Produk": product_name,
                "Skor Risiko": round(risk_score, 2),
                "Klasifikasi": risk_info["label"],
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
        2. Data kosong tidak dianalisis agar aplikasi tidak memberi klasifikasi palsu.
        3. Klasifikasi Aman, Sedang, dan Tinggi memakai satu fungsi keputusan.
        4. Gula tinggi perlu diperhatikan karena berpengaruh pada beban asupan harian.
        5. Natrium tinggi perlu dibatasi, terutama pada pengguna dengan risiko hipertensi.
        6. Lemak jenuh tinggi sebaiknya tidak dikonsumsi terlalu sering.
        7. Komposisi dengan pemanis buatan, pewarna sintetik, pengawet, dan penguat rasa menandakan indikasi produk ultra proses.
        """
    )
