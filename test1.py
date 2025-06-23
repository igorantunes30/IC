
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from skimage.feature import local_binary_pattern, hog
import joblib

# =================== CONFIGURAÇÕES ===================
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

def load_dataset(base_path):
    X, y = [], []
    for label in CLASSES:
        path = os.path.join(base_path, label)
        for file in tqdm(os.listdir(path), desc=f"Loading {label}"):
            img_path = os.path.join(path, file)
            image = cv2.imread(img_path)
            if image is None:
                continue
            image = cv2.resize(image, IMAGE_SIZE)
            feat = extract_features(image)
            X.append(feat)
            y.append(1 if label == 'MEL' else 0)
    return np.array(X), np.array(y)

# =================== CAMINHOS ===================
train_dir = "trainforclass"
val_dir = "valforclass"
test_dir = "testforclass"

# =================== CARREGAMENTO DOS DADOS ===================
X_train, y_train = load_dataset(train_dir)
X_val, y_val = load_dataset(val_dir)
X_test, y_test = load_dataset(test_dir)

# =================== NORMALIZAÇÃO ===================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_full_train = np.concatenate([X_train, X_val])
y_full_train = np.concatenate([y_train, y_val])

# =================== GRIDSEARCHCV ===================
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'class_weight': [None, 'balanced']
}

print("\n🔍 Iniciando GridSearch...")
grid = GridSearchCV(
    SVC(probability=True),
    param_grid,
    cv=3,
    scoring='f1',
    verbose=0,
    n_jobs=-1,
    return_train_score=True
)
grid.fit(X_full_train, y_full_train)

# =================== LOGS DETALHADOS POR AVALIAÇÃO ===================
print("\n📘 Logs por avaliação (estilo época):")
num_models = len(grid.cv_results_['params'])

for i in range(num_models):
    params = grid.cv_results_['params'][i]
    mean_score = grid.cv_results_['mean_test_score'][i]
    std_score = grid.cv_results_['std_test_score'][i]
    fit_time = grid.cv_results_['mean_fit_time'][i]
    rank = grid.cv_results_['rank_test_score'][i]

    print(f"""
📦 Episódio {i+1}/{num_models}
  🔧 Parâmetros: {params}
  🎯 F1 médio (validação): {mean_score:.4f} ± {std_score:.4f}
  ⏱️ Tempo médio de treino: {fit_time:.2f} segundos
  🏅 Ranking (F1): {rank}
""")

# =================== MELHOR MODELO E AVALIAÇÃO FINAL ===================
print("\n✅ Melhor modelo encontrado:")
print(grid.best_estimator_)
print("Parâmetros:", grid.best_params_)

y_pred = grid.predict(X_test)
y_proba = grid.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, target_names=["Não-MEL", "MEL"], output_dict=True)
print("\n📊 Relatório no TESTE:")
print(classification_report(y_test, y_pred, target_names=["Não-MEL", "MEL"]))
cm = confusion_matrix(y_test, y_pred)

# =================== EXPORTAÇÃO DOS RESULTADOS ===================
pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred,
    "y_proba": y_proba
}).to_excel("resultados_teste.xlsx", index=False)

pd.DataFrame(report).transpose().to_excel("relatorio_classificacao.xlsx")

# Matriz de confusão
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Não-MEL", "MEL"], yticklabels=["Não-MEL", "MEL"])
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão")
plt.tight_layout()
plt.savefig("matriz_confusao.png")
plt.show()

# Curva ROC
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
plt.savefig("curva_roc.png")
plt.show()

# Comparação entre modelos
results_df = pd.DataFrame(grid.cv_results_)
results_df[['params', 'mean_test_score', 'rank_test_score', 'mean_fit_time']].sort_values(by='rank_test_score').to_excel("gridsearch_resultados.xlsx", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    x='mean_test_score',
    y=[str(p) for p in results_df['params']],
    data=results_df,
    palette="viridis"
)
plt.xlabel("F1-score médio (validação cruzada)")
plt.ylabel("Parâmetros")
plt.title("Comparação de modelos testados no GridSearch")
plt.tight_layout()
plt.savefig("comparacao_modelos_gridsearch.png")
plt.show()

# Salvar modelo e scaler
joblib.dump(grid.best_estimator_, "modelo_svm_final.pkl")
joblib.dump(scaler, "scaler_svm_final.pkl")
print("\n💾 Modelo salvo como: modelo_svm_final.pkl")
print("💾 Scaler salvo como: scaler_svm_final.pkl")
