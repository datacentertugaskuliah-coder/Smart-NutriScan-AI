from model_utils import load_prediction_models, analyze_product_fully

def main():
    # Load the models
    feat_model, lgbm_model, w2v_model, scaler = load_prediction_models()

    if not all([feat_model, lgbm_model, w2v_model, scaler]):
        print("Failed to load one or more models. Exiting.")
        return

    # Sample data (from app.py defaults)
    nutrition_data = {
        'energi': 180, 'lemak_total': 8.0, 'lemak_jenuh': 4.0,
        'protein': 2.0, 'karbohidrat': 25.0, 'gula': 15.0,
        'garam': 0.3, 'natrium': 200
    }
    komposisi = "Tepung Terigu, Gula, Minyak Nabati, Cokelat Bubuk, Pengembang, Perisa Sintetik, Garam."

    print("--- Analyzing product ---")
    print("Nutrition data:", nutrition_data)
    print("Composition:", komposisi)
    print("-" * 20)

    # Analyze the product
    risk_score, xai_factors, recommendation = analyze_product_fully(
        nutrition_data, komposisi, feat_model, lgbm_model, w2v_model, scaler
    )

    print("--- Analysis Results ---")
    print(f"Risk Score: {risk_score:.2f}%")
    print("XAI Factors:", xai_factors)
    print("Recommendation:", recommendation)
    print("-" * 20)

if __name__ == "__main__":
    main()
