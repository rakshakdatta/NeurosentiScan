import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Restrict OpenBLAS thread usage to prevent RAM overload
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# === File Paths ===
DATA_PATH = './data/datasets/eeg_data.csv'
MODEL_DIR = './models'
EXPLAIN_HTML_DIR = './explainability/'
EXPLAIN_IMG_DIR = './explainability/lime_plots/'
os.makedirs(EXPLAIN_HTML_DIR, exist_ok=True)
os.makedirs(EXPLAIN_IMG_DIR, exist_ok=True)

# === Load Dataset ===
df = pd.read_csv(DATA_PATH)
X_raw = df.drop(columns=['label'])
y_raw = df['label']
feature_names = X_raw.columns.tolist()
class_names = sorted(y_raw.unique())

# === Config ===
N_SAMPLES = 3
models = ['SVM', 'RandomForest', 'KNN', 'NaiveBayes', 'LogisticRegression', 'DecisionTree']

# === Loop Through Each Model ===
for model_name in models:
    raw_model_path = os.path.join(MODEL_DIR, f"{model_name}_raw.joblib")
    scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.joblib")

    if not os.path.exists(raw_model_path) or not os.path.exists(scaler_path):
        print(f"❌ Skipping {model_name} — model or scaler not found")
        continue

    print(f"\n🔍 Generating LIME explanations for: {model_name}")

    try:
        model = joblib.load(raw_model_path)
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X_raw)

        X_subset = X_scaled[:N_SAMPLES]
        X_original = X_raw.iloc[:N_SAMPLES]

        explainer = LimeTabularExplainer(
            training_data=X_scaled,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )

        for i in range(N_SAMPLES):
            exp = explainer.explain_instance(
                data_row=X_subset[i],
                predict_fn=model.predict_proba,
                num_features=10
            )

            # Save as HTML
            html_path = os.path.join(EXPLAIN_HTML_DIR, f"lime_{model_name.lower()}_sample_{i+1}.html")
            exp.save_to_file(html_path)

            # Save as PNG
            fig = exp.as_pyplot_figure()
            fig.set_size_inches(10, 5)
            fig.tight_layout()
            png_path = os.path.join(EXPLAIN_IMG_DIR, f"lime_{model_name.lower()}_sample_{i+1}.png")
            plt.savefig(png_path, dpi=300)
            plt.close(fig)

        print(f"✅ {model_name}: Saved {N_SAMPLES} HTML + PNG explanations")

    except Exception as e:
        print(f"❌ Failed to explain {model_name}: {str(e)}")

print("\n🎉 All LIME explanations completed.")
