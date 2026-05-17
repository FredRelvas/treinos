"""Treino YOLO via ultralytics.

Funciona em CUDA (PC-150) e MPS (MAC). O ``device_config`` é detectado
automaticamente, mas pode ser sobrescrito por ambiente.

Uso:
    python src/train_yolo.py --model yolov8s --data data/processed/ --preset fast --subset 0.3
    python src/train_yolo.py --resume runs/exp_.../last.pt --epochs 30
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# Adiciona PC-150/ ao sys.path para resolver imports relativos.
_PC150_ROOT = Path(__file__).resolve().parents[1]
if str(_PC150_ROOT) not in sys.path:
    sys.path.insert(0, str(_PC150_ROOT))

from src.shared.augmentation import yolo_augmentation_kwargs  # noqa: E402
from src.shared.checkpoints import RunPaths, save_config, save_final_metrics  # noqa: E402
from src.shared.device_config import detect_hardware  # noqa: E402
from src.shared.presets import resolve_preset  # noqa: E402
from src.shared.utils import get_logger, set_global_seed  # noqa: E402

_log = get_logger("train_yolo")

VALID_MODELS = {"yolov8n", "yolov8s", "yolov8m"}


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treino YOLO (ultralytics).")
    p.add_argument("--model", type=str, default="yolov8s", help=f"Modelo: {sorted(VALID_MODELS)}")
    p.add_argument("--data", type=Path, help="Diretório com data/processed/ (train/val/test).")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--subset", type=float, default=None)
    p.add_argument("--preset", type=str, default="balanced", choices=["fast", "balanced", "full"])
    p.add_argument("--resume", type=Path, default=None, help="Caminho para last.pt.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", type=str, default=None, help="Nome do experimento.")
    p.add_argument("--no-augment", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
# Construção do data.yaml do ultralytics a partir do COCO JSON
# ──────────────────────────────────────────────────────────────────


def _coco_to_yolo_data_yaml(processed_root: Path, subset_fraction: float, seed: int) -> Path:
    """Converte um dataset COCO processado para o layout YOLO + data.yaml.

    O ultralytics precisa de labels em formato YOLO TXT. Geramos lado a lado
    em ``<processed_root>/_yolo/`` (idempotente, sobrescreve se existir).
    """
    from PIL import Image

    out_root = processed_root / "_yolo"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Carrega categorias do train.
    train_json = processed_root / "train" / "annotations.json"
    if not train_json.exists():
        raise FileNotFoundError(f"Esperado {train_json}. Rode prepare_dataset.py primeiro.")
    with train_json.open("r", encoding="utf-8") as fh:
        train_data = json.load(fh)
    cats = sorted(train_data["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]

    import random

    rng = random.Random(seed)

    for split in ("train", "val", "test"):
        src_img_dir = processed_root / split / "images"
        src_json = processed_root / split / "annotations.json"
        if not src_img_dir.is_dir() or not src_json.exists():
            continue
        dst_img_dir = out_root / split / "images"
        dst_lbl_dir = out_root / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        with src_json.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        imgs = {im["id"]: im for im in data["images"]}

        # Aplica subset reprodutível apenas no treino.
        all_ids = sorted(imgs.keys())
        if split == "train" and 0.0 < subset_fraction < 1.0:
            k = max(1, int(len(all_ids) * subset_fraction))
            sel = set(rng.sample(all_ids, k=k))
            ids = sorted(sel)
            _log.info("Subset train: %d/%d imagens", len(ids), len(all_ids))
        else:
            ids = all_ids

        anns_by_img: dict[int, list[dict]] = {iid: [] for iid in ids}
        sel_set = set(ids)
        for a in data["annotations"]:
            if a["image_id"] in sel_set:
                anns_by_img.setdefault(a["image_id"], []).append(a)

        for iid in ids:
            info = imgs[iid]
            src_img = src_img_dir / info["file_name"]
            if not src_img.exists():
                continue
            # symlink para não duplicar bytes.
            dst_img = dst_img_dir / info["file_name"]
            if dst_img.exists():
                dst_img.unlink()
            try:
                dst_img.symlink_to(src_img.resolve())
            except OSError:
                shutil.copy2(src_img, dst_img)

            w = info.get("width")
            h = info.get("height")
            if not w or not h:
                with Image.open(src_img) as im:
                    w, h = im.size

            lines = []
            for ann in anns_by_img.get(iid, []):
                x, y, bw, bh = ann["bbox"]
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                bwn = bw / w
                bhn = bh / h
                if bwn <= 0 or bhn <= 0:
                    continue
                cls = cat_id_to_idx[ann["category_id"]]
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bwn:.6f} {bhn:.6f}")
            (dst_lbl_dir / (Path(info["file_name"]).stem + ".txt")).write_text(
                "\n".join(lines), encoding="utf-8"
            )

    data_yaml = {
        "path": str(out_root.resolve()),
        "train": "train/images",
        "val": "val/images" if (out_root / "val" / "images").exists() else "train/images",
        "names": {i: n for i, n in enumerate(names)},
    }
    if (out_root / "test" / "images").exists():
        data_yaml["test"] = "test/images"
    yaml_path = out_root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data_yaml, fh, sort_keys=False)
    _log.info("data.yaml YOLO gerado em %s", yaml_path)
    return yaml_path


# ──────────────────────────────────────────────────────────────────
# Treino
# ──────────────────────────────────────────────────────────────────


def _load_yolo_config() -> dict:
    cfg_path = _PC150_ROOT / "configs" / "yolo_config.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _model_weights(name: str) -> str:
    if name not in VALID_MODELS:
        raise ValueError(f"Modelo inválido: {name}. Use: {sorted(VALID_MODELS)}")
    return f"{name}.pt"


def train(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    hw = detect_hardware()

    # Resume tem prioridade — herda a maioria dos parâmetros do checkpoint.
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint para resume não existe: {resume_path}")
        _log.info("Retomando treino de %s", resume_path)
        from ultralytics import YOLO

        model = YOLO(str(resume_path))
        train_kwargs = dict(resume=True)
        if args.epochs:
            train_kwargs["epochs"] = args.epochs
        model.train(**train_kwargs)
        return

    if args.data is None:
        raise SystemExit("--data é obrigatório (a menos que --resume seja usado).")

    preset = resolve_preset(
        args.preset, hw,
        epochs=args.epochs, batch=args.batch, subset=args.subset,
        no_augment=args.no_augment,
    )

    yolo_cfg = _load_yolo_config()
    data_yaml = _coco_to_yolo_data_yaml(args.data, preset.subset, args.seed)

    # Paths da run.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"yolo_{args.model}_{timestamp}"
    paths = RunPaths.new(name)

    # Augmentation: nível -> kwargs ultralytics.
    aug_kwargs = yolo_augmentation_kwargs(preset.augmentation, yolo_cfg["augmentation_overrides"])

    from ultralytics import YOLO

    model = YOLO(_model_weights(args.model))

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=preset.epochs,
        batch=preset.batch,
        imgsz=yolo_cfg.get("imgsz", 640),
        device=hw.device,
        workers=hw.num_workers,
        project=str(paths.root.parent),
        name=paths.root.name,
        exist_ok=True,
        seed=args.seed,
        optimizer=yolo_cfg.get("optimizer", "SGD"),
        lr0=yolo_cfg.get("lr0", 0.01),
        lrf=yolo_cfg.get("lrf", 0.01),
        momentum=yolo_cfg.get("momentum", 0.937),
        weight_decay=yolo_cfg.get("weight_decay", 0.0005),
        warmup_epochs=yolo_cfg.get("warmup_epochs", 3.0),
        label_smoothing=yolo_cfg.get("label_smoothing", 0.1),
        patience=yolo_cfg.get("patience", 10),
        save_period=yolo_cfg.get("save_period", 3),
        amp=hw.use_amp and yolo_cfg.get("amp", True),
        cos_lr=yolo_cfg.get("cos_lr", False),
        verbose=True,
        **aug_kwargs,
    )

    # Persiste config completa.
    save_config(paths, {
        "framework": "yolo",
        "model": args.model,
        "preset": preset.__dict__,
        "hardware": hw.__dict__,
        "args": vars(args),
        "train_kwargs": {k: str(v) for k, v in train_kwargs.items()},
    })

    _log.info("Iniciando treino: model=%s epochs=%d batch=%d device=%s",
              args.model, preset.epochs, preset.batch, hw.device)
    results = model.train(**train_kwargs)

    # ultralytics salva best.pt/last.pt dentro de paths.root/weights/. Copiamos.
    weights_dir = paths.root / "weights"
    if weights_dir.exists():
        for src_name, dst in (("best.pt", paths.best), ("last.pt", paths.last)):
            src = weights_dir / src_name
            if src.exists():
                shutil.copy2(src, dst)
        # Copia checkpoints periódicos para checkpoints/
        for ckpt in weights_dir.glob("epoch*.pt"):
            shutil.copy2(ckpt, paths.checkpoints / ckpt.name)

    # Métricas finais.
    metrics_payload = {}
    try:
        m = results.results_dict if hasattr(results, "results_dict") else {}
        metrics_payload = {str(k): float(v) for k, v in m.items() if isinstance(v, (int, float))}
    except Exception as exc:  # noqa: BLE001
        _log.warning("Falha extraindo métricas do ultralytics: %s", exc)
    save_final_metrics(paths, metrics_payload)
    _log.info("✅ Treino concluído. Run: %s", paths.root)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
