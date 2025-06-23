import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from skimage.feature import local_binary_pattern, hog
import joblib
from collections import Counter
import pandas as pd

# =================== CONFIGS ===================
IMAGE_SIZE = (128, 128)
CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
HOG_PARAMS = {'orientations': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2), 'block_norm': 'L2-Hys'}

# =================== FUNÇÕES ===================
def extract_features(image):
    features = []
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [32], [0, 256]).flatten()
        features.extend(hist)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    features.extend(lbp_hist)
    hog_feat = hog(gray, **HOG_PARAMS)
    features.extend(hog_feat)
    return np.array(features, dtype=np.float32)

def load_test_data(test_dir):
    X, y = [], []
    for label in CLASSES:
        folder = os.path.join(test_dir, label)
        for filename in os.listdir(folder):
            if filename.lower().endswith('.jpg'):
                path = os.path.join(folder, filename)
                image = cv2.imread(path)
                if image is None:
                    continue
                image = cv2.resize(image, IMAGE_SIZE)
                feat = extract_features(image)
                X.append(feat)
                y.append(1 if label == 'MEL' else 0)
    return np.array(X), np.array(y)

# =================== CAMINHOS ===================
test_dir = "valforclass"
modelo_path = "modelo_svm_final.pkl"
scaler_path = "scaler_svm_final.pkl"

# =================== CARREGAMENTO ===================
print("📂 Carregando dados de teste...")
X_test, y_test = load_test_data(test_dir)
print("✔️  Dados carregados:", Counter(y_test))

print("📦 Carregando modelo e scaler...")
modelo = joblib.load(modelo_path)
scaler = joblib.load(scaler_path)

X_test = scaler.transform(X_test)

# =================== AVALIAÇÃO ===================
print("🔍 Avaliando modelo...")
y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

# === Relatório ===
print("\n📊 Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=["Não-MEL", "MEL"]))

# === Matriz de Confusão ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Não-MEL", "MEL"], yticklabels=["Não-MEL", "MEL"])
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão")
plt.tight_layout()
plt.savefig("matriz_confusao_teste.png")
plt.show()

# === Curva ROC ===
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Falso Positivo")
plt.ylabel("Verdadeiro Positivo")
plt.title("Curva ROC - MEL vs Não-MEL")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("curva_roc_teste.png")
plt.show()

# === Resultados em Excel ===
pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "y_proba": y_proba
}).to_excel("resultados_teste_final.xlsx", index=False)

print("💾 Resultados salvos: matriz_confusao_teste.png, curva_roc_teste.png, resultados_teste_final.xlsx")
