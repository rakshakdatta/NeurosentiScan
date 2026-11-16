# scripts/load_csv_data.py

import pandas as pd
import numpy as np
import os

# 📁 Input CSV path
CSV_PATH = './data/datasets/eeg_data.csv'

# 💾 Output .npy paths
X_SAVE_PATH = './data/X_features.npy'
Y_SAVE_PATH = './data/y_labels.npy'

# 👇 Label encoding
label_map = {
    'NEGATIVE': 0,
    'NEUTRAL': 1,
    'POSITIVE': 2
}

def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV not found at {CSV_PATH}. Check the path, bhai.")
        return

    print("📥 Loading EEG CSV...")
    df = pd.read_csv(CSV_PATH)

    print(f"🔎 Dataset shape: {df.shape}")

    # 🧹 Clean label column (remove extra spaces if any)
    df['label'] = df['label'].str.strip()

    # 🎯 Encode labels
    y = df['label'].map(label_map).values
    X = df.drop(columns=['label']).values

    print(f"✅ Features shape: {X.shape}, Labels shape: {y.shape}")
    print(f"🧠 Class distribution: {np.bincount(y)}")

    # 💾 Save to .npy
    np.save(X_SAVE_PATH, X)
    np.save(Y_SAVE_PATH, y)

    print(f"📁 Saved X → {X_SAVE_PATH}")
    print(f"📁 Saved y → {Y_SAVE_PATH}")

if __name__ == '__main__':
    main()
