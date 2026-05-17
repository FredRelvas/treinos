#!/usr/bin/env bash
# Detecta ambientes disponíveis e deixa o usuário escolher.
# Use --auto para pular o menu e usar a primeira opção viável (docker → uv → pip).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[1;34m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup]${NC} $*"; }
fail()  { echo -e "${RED}[setup]${NC} $*"; }
hl()    { echo -e "${BLUE}$*${NC}"; }

AUTO=false
for arg in "$@"; do
  case "$arg" in
    --auto|-y) AUTO=true ;;
    -h|--help)
      cat <<EOF
Uso: ./setup.sh [--auto]
  --auto, -y   Não pergunta — escolhe a primeira opção viável (docker → uv → pip).
EOF
      exit 0 ;;
  esac
done

# ──────────────────────────────────────────────────────────────────
# Detecção
# ──────────────────────────────────────────────────────────────────
has_docker=false
has_nvidia=false
has_uv=false
has_python=false

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  has_docker=true
  if docker info 2>/dev/null | grep -qi 'runtimes.*nvidia\|nvidia'; then
    has_nvidia=true
  fi
fi
command -v uv      >/dev/null 2>&1 && has_uv=true
command -v python3 >/dev/null 2>&1 && has_python=true

echo
hl "═══════════════════════════════════════════════════════════"
hl "        SETUP — Pipeline de Detecção de Objetos            "
hl "═══════════════════════════════════════════════════════════"
echo
info "Ferramentas detectadas no sistema:"
$has_docker  && echo "  ✓ docker + docker compose" || echo "  ✗ docker / docker compose"
$has_nvidia  && echo "  ✓ NVIDIA runtime visível ao Docker"      || echo "  ✗ NVIDIA runtime no Docker (sem GPU passthrough)"
$has_uv      && echo "  ✓ uv ($(uv --version 2>&1 | head -1))"   || echo "  ✗ uv"
$has_python  && echo "  ✓ python3 ($(python3 --version 2>&1))"   || echo "  ✗ python3"
echo

# ──────────────────────────────────────────────────────────────────
# Opções disponíveis
# ──────────────────────────────────────────────────────────────────
declare -a OPTS
declare -a DESCS
if $has_docker && $has_nvidia; then
  OPTS+=("docker");  DESCS+=("Docker + NVIDIA — container isolado, Detectron2 incluído (~5–15 min build)")
fi
if $has_uv; then
  OPTS+=("uv");      DESCS+=("uv      — cria .venv com Python 3.11 (rápido, ~1–3 min)")
fi
if $has_python; then
  OPTS+=("pip");     DESCS+=("pip     — cria .venv com python3 padrão (~5–15 min)")
fi

if [ ${#OPTS[@]} -eq 0 ]; then
  fail "Nenhum método de instalação disponível."
  cat <<'EOF'

Instale ao menos um dos seguintes e re-execute ./setup.sh:

  • uv (recomendado, mais rápido):
      curl -LsSf https://astral.sh/uv/install.sh | sh

  • python3 + pip:
      sudo apt install python3 python3-venv python3-pip

  • Docker + nvidia-container-toolkit (Ubuntu):
      sudo apt install docker.io docker-compose-plugin nvidia-container-toolkit
EOF
  exit 1
fi

# ──────────────────────────────────────────────────────────────────
# Menu (ou modo --auto)
# ──────────────────────────────────────────────────────────────────
if $AUTO; then
  selected="${OPTS[0]}"
  info "Modo --auto: escolhendo '$selected'."
else
  hl "Escolha como instalar:"
  for i in "${!OPTS[@]}"; do
    printf "  [%d] %s\n" $((i+1)) "${DESCS[$i]}"
  done
  echo "  [q] sair"
  echo
  read -rp "Opção: " choice

  if [[ "$choice" == "q" || "$choice" == "Q" ]]; then
    info "Saindo sem instalar."
    exit 0
  fi
  if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt ${#OPTS[@]} ]; then
    fail "Opção inválida: '$choice'."
    exit 1
  fi
  selected="${OPTS[$((choice-1))]}"
fi

echo
D2_HINT_UV='uv pip install '"'"'git+https://github.com/facebookresearch/detectron2.git'"'"
D2_HINT_PIP='pip install '"'"'git+https://github.com/facebookresearch/detectron2.git'"'"
D2_HINT_MAC='MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ uv pip install '"'"'git+https://github.com/facebookresearch/detectron2.git'"'"

# ──────────────────────────────────────────────────────────────────
# Execução
# ──────────────────────────────────────────────────────────────────
case "$selected" in
  docker)
    info "Preparando container Docker (UID=$(id -u), GID=$(id -g))..."
    # Pré-cria dirs no host com tua ownership pra evitar permission denied.
    mkdir -p data/raw data/processed runs
    export HOST_UID=$(id -u)
    export HOST_GID=$(id -g)
    docker compose up -d --build
    info "Container 'cv-prova-pc150' ativo."
    echo
    hl "▶ PARA USAR:"
    echo "    docker compose exec pipeline bash"
    echo "  (dentro do shell):"
    echo "    python scripts/smoke_test.py --mode quick"
    echo
    info "Detectron2 já foi instalado durante o build (se compilação não falhou)."
    ;;

  uv)
    info "Criando .venv com uv..."
    uv venv .venv --python 3.11
    uv sync || uv pip install -r requirements.txt
    echo
    hl "▶ PARA USAR (ative o venv ANTES de rodar qualquer comando):"
    echo "    source .venv/bin/activate"
    echo "    python scripts/smoke_test.py --mode quick"
    echo
    warn "Detectron2 NÃO instalado. Para habilitar:"
    echo "    $D2_HINT_UV"
    echo "    (macOS: $D2_HINT_MAC)"
    ;;

  pip)
    info "Criando .venv com python3..."
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    echo
    hl "▶ PARA USAR (ative o venv ANTES de rodar qualquer comando):"
    echo "    source .venv/bin/activate"
    echo "    python scripts/smoke_test.py --mode quick"
    echo
    warn "Detectron2 NÃO instalado. Para habilitar:"
    echo "    $D2_HINT_PIP"
    ;;
esac

echo
info "✅ Setup concluído."
