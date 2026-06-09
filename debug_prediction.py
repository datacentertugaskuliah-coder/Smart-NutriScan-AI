#!/usr/bin/env python3
"""
Debug script untuk melihat detail prediksi model.
Jalankan dengan: python debug_prediction.py
"""

import pandas as pd
from model_utils import load_prediction_models, analyze_product_fully_debug

# Load models
print("Loading models...")
feat_model, lgbm_model, w2v_model, scaler = load_prediction_models()

if not all([feat_model, lgbm_model, w2v_model, scaler]):
    print("ERROR: Gagal memuat models!")
    exit(1)

# Test data 1: Teh Botol Sosro
print("\n" + "="*60)
print("TEST 1: Teh Botol Sosro")
print("="*60)

nutrition_data_1 = {
    'energi': 188 / 4.184,  # Convert 188Kj to kkal = 44.96
    'lemak_total': 0,
    'karbohidrat': 11,
    'gula': 9,
    'protein': 0,
    'garam': 0.01
}

composition_1 = "Air, Gula, Teh Melati (Daun Teh+Bunga Mela) (0,5%), Perisa Sintetik Bunga Melati, Penstabil."

risk_score_1, proba_1 = analyze_product_fully_debug(
    nutrition_data_1, composition_1, feat_model, lgbm_model, w2v_model, scaler
)

# Test data 2: The Kotak
print("\n" + "="*60)
print("TEST 2: The Kotak")
print("="*60)

nutrition_data_2 = {
    'energi': 70 / 4.184,  # Convert 70Kj to kkal = 16.74
    'lemak_total': 0,
    'karbohidrat': 17,
    'gula': 17,
    'protein': 0,
    'garam': 0.01
}

composition_2 = "Air, Gula, The Melati, Vitamin C"

risk_score_2, proba_2 = analyze_product_fully_debug(
    nutrition_data_2, composition_2, feat_model, lgbm_model, w2v_model, scaler
)

# Test data 3: High risk product (dummy)
print("\n" + "="*60)
print("TEST 3: High Sugar, High Fat Product (extreme)")
print("="*60)

nutrition_data_3 = {
    'energi': 400,
    'lemak_total': 25,
    'karbohidrat': 50,
    'gula': 40,
    'protein': 5,
    'garam': 1
}

composition_3 = "minyak kelapa, gula, tepung, butir es krim artificial"

risk_score_3, proba_3 = analyze_product_fully_debug(
    nutrition_data_3, composition_3, feat_model, lgbm_model, w2v_model, scaler
)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Test 1 (Teh Botol): Risk Score = {risk_score_1:.2f}%")
print(f"Test 2 (The Kotak): Risk Score = {risk_score_2:.2f}%")
print(f"Test 3 (Extreme): Risk Score = {risk_score_3:.2f}%")

if risk_score_1 == risk_score_2 == risk_score_3:
    print("\n⚠️  WARNING: All scores are identical! This suggests:")
    print("   1. Model is outputting same probability for all inputs")
    print("   2. There may be an issue with feature extraction")
    print("   3. Scaler or model weights may be problematic")
elif abs(risk_score_1 - 50) < 5 and abs(risk_score_2 - 50) < 5 and abs(risk_score_3 - 50) < 5:
    print("\n⚠️  WARNING: All scores are around 50%! This suggests:")
    print("   1. Model is predicting balanced probabilities (0.33, 0.33, 0.33)")
    print("   2. Model needs retraining with better data")
    print("   3. Feature distributions may be problematic")
else:
    print("\n✅ Scores are varied. Model seems to be working.")
