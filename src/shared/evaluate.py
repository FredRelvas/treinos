"""Cálculo de métricas de detecção (mAP via pycocotools).

Funciona em dois modos:
  - ``evaluate_coco_files``: recebe ground-truth e predições no formato COCO
    em disco e devolve um dict com mAP@0.5, mAP@0.5:0.95 e AP por classe.
  - ``predictions_to_coco_format``: converte lista de predições do nosso
    formato interno para o JSON que ``pycocotools`` aceita.
"""
from __future__ import annotations

import json
import tempfile
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Optional

from .utils import get_logger

_log = get_logger(__name__)


@dataclass
class DetectionMetrics:
    """Métricas finais de detecção."""

    map_50_95: float
    map_50: float
    ap_per_class: dict[str, float]
    precision: float
    recall: float
    avg_inference_ms: float
    num_images: int

    def to_dict(self) -> dict:
        return {
            "mAP@0.5:0.95": self.map_50_95,
            "mAP@0.5": self.map_50,
            "AP_per_class": self.ap_per_class,
            "precision": self.precision,
            "recall": self.recall,
            "avg_inference_ms": self.avg_inference_ms,
            "num_images": self.num_images,
        }


def predictions_to_coco_format(
    predictions: list[dict],
    cat_id_to_coco: Optional[dict[int, int]] = None,
) -> list[dict]:
    """Converte predições para o JSON aceito por ``COCOeval``.

    Cada predição deve ter chaves: ``image_id``, ``bbox`` (xyxy),
    ``category_id`` (id interno), ``score``.
    """
    out = []
    for p in predictions:
        x1, y1, x2, y2 = p["bbox"]
        cat_id = p["category_id"]
        if cat_id_to_coco:
            cat_id = cat_id_to_coco.get(cat_id, cat_id)
        out.append(
            {
                "image_id": int(p["image_id"]),
                "category_id": int(cat_id),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(p["score"]),
            }
        )
    return out


def evaluate_coco_files(
    gt_json: Path,
    predictions: list[dict],
    *,
    inference_times_ms: Optional[list[float]] = None,
) -> DetectionMetrics:
    """Roda COCOeval e devolve métricas consolidadas.

    Args:
        gt_json: caminho para o ``annotations.json`` do split avaliado.
        predictions: lista de predições já no formato COCO results.
        inference_times_ms: tempos individuais de inferência (opcional).
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_json))

    # COCOeval falha com lista vazia; lidamos com isso.
    if not predictions:
        _log.warning("Nenhuma predição — métricas serão zero.")
        cats = coco_gt.loadCats(coco_gt.getCatIds())
        ap_per_class = {c["name"]: 0.0 for c in cats}
        n_imgs = len(coco_gt.getImgIds())
        return DetectionMetrics(0.0, 0.0, ap_per_class, 0.0, 0.0, 0.0, n_imgs)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(predictions, fh)
        pred_path = Path(fh.name)

    try:
        coco_dt = coco_gt.loadRes(str(pred_path))
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        # Silencia o stdout barulhento do pycocotools mas mantém os números.
        with redirect_stdout(StringIO()):
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()

        stats = evaluator.stats  # 12 valores padrão COCO
        map_50_95 = float(stats[0])
        map_50 = float(stats[1])
        # AR@100 como proxy de recall global; precision via stats[0] já é mAP
        recall = float(stats[8])
        precision = map_50_95  # COCO não fornece precision global isolada

        # AP por classe (média sobre IoUs 0.5:0.95).
        cats = coco_gt.loadCats(coco_gt.getCatIds())
        ap_per_class = {}
        precisions = evaluator.eval["precision"]  # [T,R,K,A,M]
        for k, cat in enumerate(cats):
            p = precisions[:, :, k, 0, -1]
            valid = p[p > -1]
            ap_per_class[cat["name"]] = float(valid.mean()) if valid.size else 0.0
    finally:
        pred_path.unlink(missing_ok=True)

    avg_ms = (
        float(sum(inference_times_ms) / len(inference_times_ms))
        if inference_times_ms
        else 0.0
    )
    n_imgs = len(coco_gt.getImgIds())
    return DetectionMetrics(
        map_50_95=map_50_95,
        map_50=map_50,
        ap_per_class=ap_per_class,
        precision=precision,
        recall=recall,
        avg_inference_ms=avg_ms,
        num_images=n_imgs,
    )


class InferenceTimer:
    """Context manager para cronometrar inferência por imagem."""

    def __init__(self) -> None:
        self.times_ms: list[float] = []
        self._t0: float = 0.0

    def __enter__(self) -> "InferenceTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.times_ms.append((time.perf_counter() - self._t0) * 1000.0)
