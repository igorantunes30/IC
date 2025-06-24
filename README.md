## 🧐 Treinamento de Modelos de Detecção de Melanoma

## Ivan Igor Antunes Neves - 201907040042

Este projeto realiza o treinamento e avaliação de modelos de **classificação binária (MEL vs não-MEL)** utilizando **SVM (Support Vector Machines)** com engenharia de características baseadas em **histograma de cor, LBP (Local Binary Pattern)** e **HOG (Histogram of Oriented Gradients)**.

---

### 📁 Estrutura esperada do dataset

A organização dos dados é realizada com o script `modularização.py`, que estrutura os arquivos da seguinte forma:

```
/dataset
│
├── /trainforclass
│   ├── MEL
│   ├── NV
│   ├── DF
│   ├── VASC
│   ├── AKIEC
│   ├── BCC
│   └── BKL
│
├── /valforclass
│   └── (mesma estrutura do trainforclass)
│
└── /testforclass
    └── (mesma estrutura do trainforclass)
```

---

### ⚙️ Treinamento com SVM (sklearn)

O script principal para treinamento é:

```bash
Treinamento_sklearn_SVM.py
```

Ele executa os seguintes passos:

1. **Extração de características** de imagens usando cor, LBP e HOG.
2. **Padronização dos dados** com `StandardScaler`.
3. Treinamento com `GridSearchCV`, testando múltiplas combinações de hiperparâmetros (`C`, `kernel`, `gamma`, `class_weight`) para encontrar a melhor configuração com base no **F1-score**.
4. Uso de **validação cruzada** para generalização eficaz.
5. **Avaliação final** com o conjunto `testforclass` para reportar as métricas.

---

### 📦 Saídas geradas

Após o treinamento:

* ✅ Modelo salvo em: `modelo_svm_final.pkl`
* ✅ Scaler salvo em: `scaler_svm_final.pkl`
* 📊 Métricas do teste exportadas para: `relatorio_classificacao.xlsx`
* 📁 Previsões do teste salvas em: `resultados_teste.xlsx`
* 📉 Gráficos:

  * `matriz_confusao.png` — Matriz de confusão
  * `curva_roc.png` — Curva ROC (Área sob a curva)
  * `comparacao_modelos_gridsearch.png` — Comparação entre modelos testados no GridSearch
* 📈 Tabela com desempenho de todos os modelos testados: `gridsearch_resultados.xlsx`

---

### 🔄 Validação dinâmica

Durante o `GridSearchCV`, utilizamos uma estratégia de **validação cruzada com troca dinâmica dos conjuntos** para mitigar overfitting e melhorar a capacidade de generalização.

O conjunto `testforclass` **não é usado no treinamento** — ele é reservado exclusivamente para a **avaliação final**.

---

### 🥺 Pós-análise

O script `verification.py` pode ser utilizado para **análise complementar dos dados** e das classificações, com base na estrutura gerada por `modularização.py`.
