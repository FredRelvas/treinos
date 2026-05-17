"""Utilitários globais: seed, logging e resolução de paths.

Centraliza configurações que precisam ser consistentes em toda a pipeline
para garantir reprodutibilidade e mensagens uniformes.
"""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────


def project_root() -> Path:
    """Retorna a raiz do projeto PC-150 (onde fica `pyproject.toml`).

    Sobe na árvore a partir deste arquivo até encontrar `pyproject.toml`.
    Funciona tanto chamado de PC-150/ quanto de MAC/ (caminho relativo).
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume estrutura PC-150/src/shared/utils.py
    return here.parents[2]


def runs_dir() -> Path:
    """Retorna o diretório `runs/` (cria se não existir)."""
    path = project_root() / "runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path) -> Path:
    """Garante que `path` exista como diretório e retorna o próprio Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ──────────────────────────────────────────────────────────────────
# Seed
# ──────────────────────────────────────────────────────────────────


def set_global_seed(seed: int = 42) -> None:
    """Define seed em random, numpy, torch (CPU/CUDA) e ``PYTHONHASHSEED``."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # MPS herda da seed CPU (torch.manual_seed)
    except ImportError:
        # PyTorch ainda não disponível (durante setup) — não é erro.
        pass


# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────

_LOG_FMT = "[%(asctime)s] %(levelname)s %(name)s — %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "pipeline", level: Optional[int] = None) -> logging.Logger:
    """Retorna um logger configurado com formato uniforme em stderr.

    Idempotente: chamadas repetidas não duplicam handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
        logger.addHandler(handler)
        logger.propagate = False
    if level is None:
        level = logging.DEBUG if os.environ.get("PIPELINE_DEBUG") else logging.INFO
    logger.setLevel(level)
    return logger
