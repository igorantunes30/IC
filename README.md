## Treinamento de modelos de detecção de melanoma
O arquivo Treinamento_sklearn_SVM.py treina o modelo de sklearn.svm
Os arquivos precisão ser reorganizados com o codigo da seguinte forma
/dataset
    /trainforclass
        /MEL
        /NV
        /DF
        /VASC
        /AKIEC
        /BCC
        /BKL
    /valforclass
        /MEL
        /NV
        /DF
        /VASC
        /AKIEC
        /BCC
        /BKL
    /testforclass
        /MEL
        /NV
        /DF
        /VASC
        /AKIEC
        /BCC
        /BKL
Usei um metodo de troca de validação a cada interação na tentativa de generalizar de forma eficiente
