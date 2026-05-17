"""Gerenciamento de runs/ e checkpoints.

Cada treino abre uma pasta única ``runs/exp_{TIMESTAMP}_{NOME}/`` com a
estrutura definida no escopo (checkpoints/, best.pt, last.pt, config.json,
metrics.json, metrics_per_epoch.csv, plots/).
"""
from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .utils import ensure_dir, get_logger, runs_dir

_log = get_logger(__name__)

SAVE_EVERY_N_EPOCHS = 3  # política fixa


@dataclass
class RunPaths:
    """Conjunto de caminhos canônicos de uma run."""

    root: Path
    checkpoints: Path
    plots: Path
    inference_samples: Path
    best: Path
    last: Path
    config: Path
    metrics: Path
    metrics_csv: Path

    @classmethod
    def new(cls, name: Optional[str] = None) -> "RunPaths":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = (name or "run").strip().replace(" ", "_")
        root = ensure_dir(runs_dir() / f"exp_{ts}_{slug}")
        return cls.from_root(root)

    @classmethod
    def from_root(cls, root: Path) -> "RunPaths":
        root = Path(root)
        return cls(
            root=root,
            checkpoints=ensure_dir(root / "checkpoints"),
            plots=ensure_dir(root / "plots"),
            inference_samples=ensure_dir(root / "plots" / "inference_samples"),
            best=root / "best.pt",
            last=root / "last.pt",
            config=root / "config.json",
            metrics=root / "metrics.json",
            metrics_csv=root / "metrics_per_epoch.csv",
        )


# ──────────────────────────────────────────────────────────────────
# Persistência
# ──────────────────────────────────────────────────────────────────


def save_config(paths: RunPaths, config: dict) -> None:
    """Grava ``config.json`` com todos os hiperparâmetros usados."""
    with paths.config.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, default=str)
    _log.info("Config salva em %s", paths.config)


def save_final_metrics(paths: RunPaths, metrics: dict) -> None:
    with paths.metrics.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)


def append_epoch_metrics(paths: RunPaths, row: dict) -> None:
    """Acrescenta uma linha ao CSV de métricas por época (cria header se vazio)."""
    write_header = not paths.metrics_csv.exists()
    with paths.metrics_csv.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ──────────────────────────────────────────────────────────────────
# Política de checkpoints
# ──────────────────────────────────────────────────────────────────


@dataclass
class CheckpointTracker:
    """Decide quando salvar checkpoints periódicos / best / last.

    Não copia arquivos sozinho; o código de treino chama os métodos
    ``should_save_periodic`` e ``update_best`` e fornece o source.
    """

    paths: RunPaths
    best_metric: float = -1.0
    patience: int = 10
    epochs_without_improvement: int = 0
    save_every: int = SAVE_EVERY_N_EPOCHS

    def should_save_periodic(self, epoch: int) -> bool:
        return epoch > 0 and epoch % self.save_every == 0

    def epoch_ckpt_path(self, epoch: int) -> Path:
        return self.paths.checkpoints / f"epoch_{epoch:03d}.pt"

    def update_best(self, current_metric: float, source: Path) -> bool:
        """Copia ``source`` para ``best.pt`` se a métrica melhorou.

        Returns:
            True se best foi atualizado.
        """
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.epochs_without_improvement = 0
            if source.exists() and source.resolve() != self.paths.best.resolve():
                shutil.copy2(source, self.paths.best)
            _log.info(
                "🏆 Novo best: %.4f → %s", current_metric, self.paths.best.name
            )
            return True
        self.epochs_without_improvement += 1
        return False

    def update_last(self, source: Path) -> None:
        if source.exists() and source.resolve() != self.paths.last.resolve():
            shutil.copy2(source, self.paths.last)

    def should_early_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience
