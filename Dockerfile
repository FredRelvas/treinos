# Imagem base: PyTorch 2.3 + CUDA 12.1 + cuDNN 8 (compatível com RTX 4090).
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

# Dependências de SO necessárias para OpenCV, build de Detectron2 e ferramentas comuns.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Instala uv (gerenciador de dependências moderno e rápido).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv

# Usuário não-root (UID 1000).
RUN useradd -m -u 1000 -s /bin/bash app
WORKDIR /workspace

# Copia manifests primeiro para aproveitar cache do Docker.
COPY --chown=app:app pyproject.toml requirements.txt ./

# Instala dependências base via uv (resolve PyTorch já presente na imagem).
RUN uv pip install --system -r requirements.txt

# Instala Detectron2 da master branch (compilação contra CUDA da imagem).
RUN uv pip install --system 'git+https://github.com/facebookresearch/detectron2.git' \
    || (echo "⚠️  Detectron2 falhou no build; instale manualmente depois." && true)

# Copia o restante do código.
COPY --chown=app:app . .

# Garante que runs/ exista e seja escrita pelo usuário app.
RUN mkdir -p /workspace/runs /workspace/data && chown -R app:app /workspace

USER app

EXPOSE 8888
CMD ["bash"]
