FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Configurar diretório de trabalho
WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivos do projeto
COPY . /app/

# Instalar dependências Python
RUN pip3 install --no-cache-dir \
    numpy \
    Pillow \
    opencv-python \
    torchvision \
    tensorboard

# Criar diretórios necessários
RUN mkdir -p /app/log /app/ckpt /app/resultados

# Definir variáveis de ambiente
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Comando padrão para executar o teste em lote
ENTRYPOINT ["python3", "test_batch.py"]
CMD ["--help"] 