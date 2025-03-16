#!/bin/bash

# Verifica se o container já existe
CONTAINER_NAME="p2pnet"
if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=$CONTAINER_NAME)" ]; then
        # Remove o container existente
        docker rm $CONTAINER_NAME
    fi
fi

# Constrói a imagem se necessário
docker build -t p2pnet .

# Executa o container com os volumes necessários
docker run --gpus all \
    --name $CONTAINER_NAME \
    -v "$(pwd)/ckpt:/app/ckpt" \
    -v "$(pwd)/resultados:/app/resultados" \
    -v "$(pwd)/input:/app/input" \
    p2pnet \
    --input_dir /app/input \
    --output_dir /app/resultados \
    --weight_path /app/ckpt/latest.pth \
    --confidence_threshold 0.3 \
    --multi_scale \
    "$@" 