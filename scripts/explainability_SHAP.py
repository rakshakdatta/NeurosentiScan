import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D
from keras.utils import to_categorical

# ========== Suppress SHAP Matplotlib Warning ==========
warnings.filterwarnings("ignore", category=UserWarning)

# ========== Load Sample EEG-like Dataset (Iris for demo) ==========
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
feature_names = X.columns.tolist()

# ========== Preprocessing ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== SHAP for ML Model (Random Forest) ==========
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_ml, y_train_ml)

explainer_rf = shap.TreeExplainer(rf)
shap_values_rf = explainer_rf.shap_values(X_test_ml)

shap.summary_plot(shap_values_rf, X_test_ml, feature_names=feature_names, cmap='coolwarm', show=False)
plt.title("SHAP Summary - Random Forest (ML)")
plt.tight_layout()
plt.savefig("shap_rf_summary.png", dpi=300)
plt.clf()

# ========== SHAP for DL Model (1D CNN) ==========
X_dl = X_scaled.reshape(-1, 2, 2)  # Dummy reshape for CNN
y_dl = to_categorical(y)

X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_dl, y_dl, test_size=0.2, random_state=42)

model = Sequential([
    Conv1D(8, kernel_size=1, activation='relu', input_shape=(2, 2)),
    Flatten(),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_dl, y_train_dl, epochs=10, verbose=0)

# Flattened data for SHAP KernelExplainer
background = X_train_dl[:50].reshape(50, -1)
test_samples = X_test_dl[:10].reshape(10, -1)

explainer_dl = shap.KernelExplainer(lambda x: model.predict(x.reshape(-1, 2, 2)), background)
shap_values_dl = explainer_dl.shap_values(test_samples)

shap.summary_plot(shap_values_dl[0], test_samples, feature_names=[f"f{i}" for i in range(test_samples.shape[1])], cmap='coolwarm', show=False)
plt.title("SHAP Summary - CNN (DL)")
plt.tight_layout()
plt.savefig("shap_dl_summary.png", dpi=300)
plt.clf()
