"""Plots da pipeline: curvas de loss/mAP, PR, matriz de confusão e amostras.

Todas as funções usam matplotlib em modo non-interactive (Agg).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")  # garante backend sem display (Docker/SSH)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .utils import ensure_dir, get_logger  # noqa: E402

_log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────
# Curvas de treino
# ──────────────────────────────────────────────────────────────────


def plot_loss_curve(epochs: list[int], train_loss: list[float], val_loss: Optional[list[float]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, label="train", marker="o")
    if val_loss and any(v is not None for v in val_loss):
        ax.plot(epochs, val_loss, label="val", marker="s")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_title("Curva de Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_map_evolution(epochs: list[int], map50: list[float], map5095: Optional[list[float]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, map50, label="mAP@0.5", marker="o")
    if map5095:
        ax.plot(epochs, map5095, label="mAP@0.5:0.95", marker="s")
    ax.set_xlabel("Época")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 1)
    ax.set_title("Evolução do mAP")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_precision_recall(pr_curves: dict[str, tuple[list[float], list[float]]], out: Path) -> Path:
    """Plota curvas PR por classe. ``pr_curves`` = {class_name: (recall, precision)}."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (recall, precision) in pr_curves.items():
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision × Recall por classe")
    ax.grid(True, alpha=0.3)
    if pr_curves:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(class_names), 1 + 0.6 * len(class_names)))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ──────────────────────────────────────────────────────────────────
# Amostras de inferência
# ──────────────────────────────────────────────────────────────────


def draw_predictions_vs_gt(
    image_path: Path,
    predictions: list[dict],
    ground_truth: list[dict],
    class_names: list[str],
    out: Path,
) -> Path:
    """Salva uma figura com a imagem + bboxes preditas (verde) e GT (vermelho).

    Cada bbox em ``predictions`` deve ter: ``bbox`` (xyxy), ``category_id``, ``score``.
    Cada bbox em ``ground_truth`` deve ter: ``bbox`` (xyxy), ``category_id``.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)
    ax.axis("off")

    for gt in ground_truth:
        x1, y1, x2, y2 = gt["bbox"]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
        name = class_names[gt["category_id"]] if gt["category_id"] < len(class_names) else "?"
        ax.text(x1, max(0, y1 - 4), f"GT:{name}", color="red", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))

    for pred in predictions:
        x1, y1, x2, y2 = pred["bbox"]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", linewidth=2))
        name = class_names[pred["category_id"]] if pred["category_id"] < len(class_names) else "?"
        score = pred.get("score", 0.0)
        ax.text(x1, y2 + 12, f"{name} {score:.2f}", color="lime", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1))

    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def save_inference_samples(
    samples: Iterable[tuple[Path, list[dict], list[dict]]],
    class_names: list[str],
    out_dir: Path,
    max_samples: int = 10,
) -> list[Path]:
    """Salva até ``max_samples`` figuras de amostras anotadas."""
    out_dir = ensure_dir(out_dir)
    saved = []
    for i, (img_path, preds, gts) in enumerate(samples):
        if i >= max_samples:
            break
        out = out_dir / f"sample_{i + 1:03d}.png"
        draw_predictions_vs_gt(img_path, preds, gts, class_names, out)
        saved.append(out)
    _log.info("Salvou %d amostras de inferência em %s", len(saved), out_dir)
    return saved
