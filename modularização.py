import os
import shutil
import pandas as pd

# Caminhos
caminho_csv = 'ISIC2018_Task3_Test_GroundTruth.csv'
pasta_origem = r'ISIC2018_Task3_Test_Input\ISIC2018_Task3_Test_Input'
pasta_destino_base = 'testforclass'

# Carregar o CSV
df = pd.read_csv("ISIC2018_Task3_Test_GroundTruth.csv")

# Classes disponíveis
classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# Criar pastas por classe
for classe in classes:
    pasta_classe = os.path.join(pasta_destino_base, classe)
    os.makedirs(pasta_classe, exist_ok=True)

# Para cada imagem, verificar a classe e copiar
for _, row in df.iterrows():
    nome_imagem = row['image'] + '.jpg'
    caminho_imagem = os.path.join(pasta_origem, nome_imagem)

    for classe in classes:
        if row[classe] == 1.0:
            destino = os.path.join(pasta_destino_base, classe, nome_imagem)
            if os.path.exists(caminho_imagem):
                print(f"Copiando imagem: {caminho_imagem} → {destino}")
                shutil.copy2(caminho_imagem, destino)
            else:
                print(f"[ERRO] Imagem não encontrada: {caminho_imagem}")
            break

print("Organização do conjunto de teste concluída.")
