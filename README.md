## Treinamento de modelos de detecção de melanoma
O arquivo Treinamento_sklearn_SVM.py treina o modelo de sklearn.svm

Os arquivos precisão ser reorganizados, com o codigo modularização.py, da seguinte forma

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
        
Usei um metodo de troca de validação a cada interação na tentativa de generalizar de forma eficiente.

Depois do treinamento usei o testforclass para obter os resultados 

Afim de obter a melhor combinação de parametros ajustaveis do sklearn, fiz um laço para que ele modificasse diversas vezes. Quando ele achar a melhor combinação, vai gerar o modelo modelo_svm_final.pkl e scaler_svm_final.pkl

É gerado tambem arquivos csv com o relatório das metricas(relatorio_classificacao.xlsx) e resultados(resultados_teste.xlsx) para todas as combinações de parametros

O verification.py serve para analise de dados depois da reorganização do modularização.py

