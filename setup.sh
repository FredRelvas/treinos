#!/usr/bin/env bash
# Detecta o ambiente disponível e configura a pipeline.
# Ordem: Docker (+ nvidia) → uv → pip → instruções manuais.
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup]${NC} $*"; }
fail()  { echo -e "${RED}[setup]${NC} $*"; }

# Move para o diretório do script.
cd "$(dirname "${BASH_SOURCE[0]}")"

D2_HINT_UV='uv pip install '"'"'git+https://github.com/facebookresearch/detectron2.git'"'"
D2_HINT_PIP='pip install '"'"'git+https://github.com/facebookresearch/detectron2.git'"'"
D2_HINT_MAC='MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ uv pip install '"'"'git+https://github.com/facebookresearch/detectron2.git'"'"

print_d2_hint() {
  echo
  warn "Detectron2 NÃO é instalado automaticamente. Para habilitar:"
  echo "  $1"
  echo "  (no macOS, use: $D2_HINT_MAC)"
  echo
}

# ──────────────────────────────────────────────────────────────────
# 1) Docker + nvidia-container-toolkit
# ──────────────────────────────────────────────────────────────────
if command -v docker >/dev/null 2>&1 && command -v docker compose >/dev/null 2>&1; then
  if docker info 2>/dev/null | grep -q -i nvidia; then
    info "Docker + NVIDIA detectados — subindo container."
    docker compose up -d --build
    info "Container ativo. Entre com: docker compose exec pipeline bash"
    exit 0
  else
    warn "Docker presente, mas runtime NVIDIA não detectado. Pulando Docker."
  fi
fi

# ──────────────────────────────────────────────────────────────────
# 2) uv
# ──────────────────────────────────────────────────────────────────
if command -v uv >/dev/null 2>&1; then
  info "uv detectado — criando venv e instalando dependências."
  uv venv .venv --python 3.11
  uv sync || uv pip install -r requirements.txt
  info "Ambiente pronto. Ative com: source .venv/bin/activate"
  print_d2_hint "$D2_HINT_UV"
  exit 0
fi

# ──────────────────────────────────────────────────────────────────
# 3) pip (fallback)
# ──────────────────────────────────────────────────────────────────
if command -v python3 >/dev/null 2>&1; then
  info "uv ausente — usando pip + venv padrão."
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  info "Ambiente pronto. Ative com: source .venv/bin/activate"
  print_d2_hint "$D2_HINT_PIP"
  exit 0
fi

# ──────────────────────────────────────────────────────────────────
# 4) Instruções manuais
# ──────────────────────────────────────────────────────────────────
fail "Nenhum dos seguintes foi encontrado: docker, uv, python3."
cat <<'EOF'

Instale manualmente:

  Opção A (recomendada): uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv .venv --python 3.11
    uv sync
    uv pip install 'git+https://github.com/facebookresearch/detectron2.git'

  Opção B: pip puro
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install 'git+https://github.com/facebookresearch/detectron2.git'

  Opção C: Docker (NVIDIA host)
    docker compose up -d --build

EOF
exit 1
