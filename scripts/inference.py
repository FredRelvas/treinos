"""Inferência final + cálculo de métricas + geração de plots/relatórios.

Funciona com pesos YOLO (--framework yolo) ou Detectron2 (--framework detectron2).
Saídas em ``runs/exp_X/inference/``:
  - metrics.json
  - inference_report.json (predições + GT por imagem)
  - plots/precision_recall.png, confusion_matrix.png
  - plots/inference_samples/*.png (≥10 imagens anotadas)

Uso:
    python scripts/inference.py \
        --model-path runs/exp_.../best.pt \
        --data data/processed/test \
        --framework yolo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

_PC150_ROOT = Path(__file__).resolve().parents[1]
if str(_PC150_ROOT) not in sys.path:
    sys.path.insert(0, str(_PC150_ROOT))

from src.shared.evaluate import (  # noqa: E402
    InferenceTimer,
    evaluate_coco_files,
    predictions_to_coco_format,
)
from src.shared.utils import ensure_dir, get_logger, set_global_seed  # noqa: E402
from src.shared.visualize import (  # noqa: E402
    plot_confusion_matrix,
    plot_precision_recall,
    save_inference_samples,
)

_log = get_logger("inference")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inferência + métricas finais.")
    p.add_argument("--model-path", type=Path, required=True, help="Pesos treinados.")
    p.add_argument("--data", type=Path, required=True,
                   help="Split a avaliar (ex: data/processed/test ou val).")
    p.add_argument("--framework", choices=["yolo", "detectron2"], required=True)
    p.add_argument("--conf", type=float, default=0.25, help="Threshold de confiança.")
    p.add_argument("--iou", type=float, default=0.5, help="IoU NMS.")
    p.add_argument("--max-samples", type=int, default=10, help="Imagens anotadas a salvar.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
# Carregamento do split
# ──────────────────────────────────────────────────────────────────


def _resolve_split(data_path: Path) -> tuple[Path, Path]:
    """Retorna (images_dir, annotations.json). Aceita data/processed/test ou test/."""
    if (data_path / "annotations.json").exists() and (data_path / "images").is_dir():
        return data_path / "images", data_path / "annotations.json"
    # talvez seja data/processed (raiz); preferimos test, fallback val.
    for name in ("test", "val", "train"):
        sub = data_path / name
        if (sub / "annotations.json").exists() and (sub / "images").is_dir():
            return sub / "images", sub / "annotations.json"
    raise FileNotFoundError(f"Split COCO não encontrado em {data_path}")


# ──────────────────────────────────────────────────────────────────
# Predição com YOLO
# ──────────────────────────────────────────────────────────────────


def _predict_yolo(
    model_path: Path,
    images_dir: Path,
    gt: dict,
    conf: float,
    iou: float,
) -> tuple[list[dict], list[dict], list[float]]:
    """Retorna (predictions_internas, predictions_coco_results, tempos_ms)."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    cat_ids = sorted({c["id"] for c in gt["categories"]})
    idx_to_coco_id = {i: cid for i, cid in enumerate(cat_ids)}

    internal: list[dict] = []
    coco_results: list[dict] = []
    times: list[float] = []

    images_by_filename = {im["file_name"]: im for im in gt["images"]}

    for img_info in gt["images"]:
        path = images_dir / img_info["file_name"]
        if not path.exists():
            continue
        with InferenceTimer() as t:
            res = model.predict(source=str(path), conf=conf, iou=iou, verbose=False)
        times.extend(t.times_ms)

        if not res:
            continue
        r = res[0]
        if r.boxes is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for b, s, c in zip(boxes, scores, cls):
            pred = {
                "image_id": img_info["id"],
                "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "category_id": int(c),
                "score": float(s),
                "image_file": img_info["file_name"],
            }
            internal.append(pred)
            coco_results.append({
                "image_id": img_info["id"],
                "category_id": int(idx_to_coco_id.get(int(c), int(c) + 1)),
                "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                "score": float(s),
            })

    return internal, coco_results, times


# ──────────────────────────────────────────────────────────────────
# Predição com Detectron2
# ──────────────────────────────────────────────────────────────────


def _predict_detectron2(
    model_path: Path,
    images_dir: Path,
    gt: dict,
    conf: float,
) -> tuple[list[dict], list[dict], list[float]]:
    try:
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
    except ImportError as exc:
        raise SystemExit(f"Detectron2 não instalado: {exc}")

    import cv2

    cfg = get_cfg()
    # Usa Faster R-CNN como base — ajuste se for outro arquitetura.
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf
    cfg.MODEL.DEVICE = "cuda" if _cuda_available() else "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(gt["categories"])
    predictor = DefaultPredictor(cfg)

    cat_ids = sorted({c["id"] for c in gt["categories"]})
    idx_to_coco_id = {i: cid for i, cid in enumerate(cat_ids)}

    internal: list[dict] = []
    coco_results: list[dict] = []
    times: list[float] = []

    for img_info in gt["images"]:
        path = images_dir / img_info["file_name"]
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        with InferenceTimer() as t:
            out = predictor(img)
        times.extend(t.times_ms)
        inst = out["instances"].to("cpu")
        boxes = inst.pred_boxes.tensor.numpy()
        scores = inst.scores.numpy()
        cls = inst.pred_classes.numpy().astype(int)
        for b, s, c in zip(boxes, scores, cls):
            pred = {
                "image_id": img_info["id"],
                "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "category_id": int(c),
                "score": float(s),
                "image_file": img_info["file_name"],
            }
            internal.append(pred)
            coco_results.append({
                "image_id": img_info["id"],
                "category_id": int(idx_to_coco_id.get(int(c), int(c) + 1)),
                "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                "score": float(s),
            })

    return internal, coco_results, times


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────
# Confusion matrix simples (greedy IoU)
# ──────────────────────────────────────────────────────────────────


def _build_confusion_matrix(
    gt: dict, preds: list[dict], iou_thr: float = 0.5
) -> np.ndarray:
    cats = sorted(gt["categories"], key=lambda c: c["id"])
    n = len(cats)
    # +1 para background (FP/FN)
    matrix = np.zeros((n + 1, n + 1), dtype=int)
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    preds_by_img: dict[int, list[dict]] = {}
    for p in preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)

    gts_by_img: dict[int, list[dict]] = {}
    for a in gt["annotations"]:
        gts_by_img.setdefault(a["image_id"], []).append(a)

    def iou(b1, b2):
        x1, y1, x2, y2 = b1
        gx, gy, gw, gh = b2
        gx2, gy2 = gx + gw, gy + gh
        ix1, iy1 = max(x1, gx), max(y1, gy)
        ix2, iy2 = min(x2, gx2), min(y2, gy2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        a1 = max(0.0, (x2 - x1) * (y2 - y1))
        a2 = max(0.0, gw * gh)
        u = a1 + a2 - inter
        return inter / u if u > 0 else 0.0

    for img_id in set(list(preds_by_img) + list(gts_by_img)):
        gts = list(gts_by_img.get(img_id, []))
        ps = sorted(preds_by_img.get(img_id, []), key=lambda p: -p["score"])
        matched = set()
        for p in ps:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if j in matched:
                    continue
                v = iou(p["bbox"], g["bbox"])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_thr and best_j >= 0:
                gt_idx = cat_id_to_idx[gts[best_j]["category_id"]]
                matrix[gt_idx, p["category_id"]] += 1
                matched.add(best_j)
            else:
                matrix[n, p["category_id"]] += 1  # FP
        for j, g in enumerate(gts):
            if j not in matched:
                gt_idx = cat_id_to_idx[g["category_id"]]
                matrix[gt_idx, n] += 1  # FN
    return matrix


# ──────────────────────────────────────────────────────────────────
# PR curves baseadas nas predições (não usa COCOeval para evitar parsing complexo)
# ──────────────────────────────────────────────────────────────────


def _build_pr_curves(gt: dict, preds: list[dict], iou_thr: float = 0.5) -> dict:
    cats = sorted(gt["categories"], key=lambda c: c["id"])
    curves: dict[str, tuple[list[float], list[float]]] = {}
    preds_by_img: dict[int, list[dict]] = {}
    for p in preds:
        preds_by_img.setdefault(p["image_id"], []).append(p)
    gts_by_img: dict[int, list[dict]] = {}
    for a in gt["annotations"]:
        gts_by_img.setdefault(a["image_id"], []).append(a)
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}

    def iou(b1, b2):
        x1, y1, x2, y2 = b1
        gx, gy, gw, gh = b2
        gx2, gy2 = gx + gw, gy + gh
        ix1, iy1 = max(x1, gx), max(y1, gy)
        ix2, iy2 = min(x2, gx2), min(y2, gy2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        u = max(0.0, (x2 - x1) * (y2 - y1)) + max(0.0, gw * gh) - inter
        return inter / u if u > 0 else 0.0

    for cat in cats:
        cls_idx = cat_id_to_idx[cat["id"]]
        # Coleta todas as predições da classe.
        all_preds = []
        n_gt = 0
        for img_id, gts in gts_by_img.items():
            n_gt += sum(1 for g in gts if g["category_id"] == cat["id"])
        for img_id, ps in preds_by_img.items():
            for p in ps:
                if p["category_id"] != cls_idx:
                    continue
                # Match com gts dessa classe naquela imagem.
                cls_gts = [g for g in gts_by_img.get(img_id, []) if g["category_id"] == cat["id"]]
                best = max((iou(p["bbox"], g["bbox"]) for g in cls_gts), default=0.0)
                all_preds.append((p["score"], 1 if best >= iou_thr else 0))

        if n_gt == 0 or not all_preds:
            curves[cat["name"]] = ([0.0, 1.0], [0.0, 0.0])
            continue
        all_preds.sort(key=lambda x: -x[0])
        tp = 0
        fp = 0
        precisions, recalls = [], []
        for _, is_tp in all_preds:
            if is_tp:
                tp += 1
            else:
                fp += 1
            precisions.append(tp / max(1, tp + fp))
            recalls.append(tp / n_gt)
        curves[cat["name"]] = (recalls, precisions)
    return curves


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    images_dir, ann_path = _resolve_split(args.data)
    _log.info("Avaliando %s sobre %s", args.model_path, ann_path)
    with ann_path.open("r", encoding="utf-8") as fh:
        gt = json.load(fh)

    if args.framework == "yolo":
        internal, coco_results, times = _predict_yolo(args.model_path, images_dir, gt, args.conf, args.iou)
    else:
        internal, coco_results, times = _predict_detectron2(args.model_path, images_dir, gt, args.conf)

    # Saídas vão para a pasta da run (parent do model_path se for runs/exp_X/best.pt).
    run_root = args.model_path.parent
    out_root = ensure_dir(run_root / "inference")
    plots_dir = ensure_dir(out_root / "plots")
    samples_dir = ensure_dir(plots_dir / "inference_samples")

    # Métricas via COCOeval.
    metrics = evaluate_coco_files(ann_path, coco_results, inference_times_ms=times)
    with (out_root / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics.to_dict(), fh, indent=2)

    # Relatório por imagem.
    by_image: dict[int, dict] = {}
    for im in gt["images"]:
        by_image[im["id"]] = {
            "file_name": im["file_name"],
            "predictions": [],
            "ground_truth": [],
        }
    for p in internal:
        by_image[p["image_id"]]["predictions"].append({
            "bbox": p["bbox"], "category_id": p["category_id"], "score": p["score"],
        })
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(sorted(gt["categories"], key=lambda c: c["id"]))}
    for a in gt["annotations"]:
        x, y, w, h = a["bbox"]
        by_image[a["image_id"]]["ground_truth"].append({
            "bbox": [x, y, x + w, y + h],
            "category_id": cat_id_to_idx[a["category_id"]],
        })
    with (out_root / "inference_report.json").open("w", encoding="utf-8") as fh:
        json.dump(by_image, fh, indent=2)

    # Plots.
    class_names = [c["name"] for c in sorted(gt["categories"], key=lambda c: c["id"])]
    pr_curves = _build_pr_curves(gt, internal)
    plot_precision_recall(pr_curves, plots_dir / "precision_recall.png")
    cm = _build_confusion_matrix(gt, internal)
    plot_confusion_matrix(cm, class_names + ["background"], plots_dir / "confusion_matrix.png")

    # Amostras anotadas.
    samples = []
    for img_id, info in by_image.items():
        samples.append((images_dir / info["file_name"], info["predictions"], info["ground_truth"]))
    save_inference_samples(samples, class_names, samples_dir, max_samples=args.max_samples)

    # Copia curvas de treino se existirem.
    for src_name in ("loss_curve.png", "map_evolution.png"):
        src = run_root / "plots" / src_name
        if src.exists():
            import shutil

            shutil.copy2(src, plots_dir / src_name)

    print("\n" + "=" * 60)
    print("RESULTADOS DE INFERÊNCIA")
    print("-" * 60)
    print(f"  mAP@0.5:        {metrics.map_50:.4f}")
    print(f"  mAP@0.5:0.95:   {metrics.map_50_95:.4f}")
    print(f"  Precision:      {metrics.precision:.4f}")
    print(f"  Recall:         {metrics.recall:.4f}")
    print(f"  Tempo médio:    {metrics.avg_inference_ms:.2f} ms/imagem")
    print(f"  Imagens:        {metrics.num_images}")
    print("  AP por classe:")
    for name, ap in metrics.ap_per_class.items():
        print(f"    {name:<25} {ap:.4f}")
    print("=" * 60)
    print(f"\nSaídas em {out_root}")


if __name__ == "__main__":
    main()
