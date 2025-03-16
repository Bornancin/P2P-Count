import kagglehub
import os
import shutil

# Download latest version
print("Baixando o conjunto de dados...")
path = kagglehub.dataset_download("hosammhmdali/shanghai-tech-dataset-part-a-and-part-b")

print("Caminho para os arquivos do dataset:", path)
print("Conteúdo do diretório:")
for file in os.listdir(path):
    print(f"- {file}")

# Copiar os arquivos para o diretório correto
print("\nCopiando os arquivos...")
src_dir = os.path.join(path, "ShanghaiTech")
if os.path.exists("new_public_density_data"):
    shutil.rmtree("new_public_density_data")
shutil.copytree(src_dir, "new_public_density_data")

print("\nDataset copiado com sucesso!")
print("\nConteúdo do diretório new_public_density_data:")
for root, dirs, files in os.walk("new_public_density_data"):
    level = root.replace("new_public_density_data", "").count(os.sep)
    indent = " " * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 4 * (level + 1)
    for f in files:
        print(f"{subindent}{f}") 