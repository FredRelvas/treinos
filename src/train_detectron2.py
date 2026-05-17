"""Treino Detectron2 com early stopping e logging customizado.

O Detectron2 é OPCIONAL — se o import falhar, exibe instruções claras
e encerra com código 2 (ambiente recuperável).

Uso:
    python src/train_detectron2.py --model faster_rcnn --data data/processed/ --preset balanced
    python src/train_detectron2.py --model retinanet --data data/processed/ --preset full --name d2_run
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

_PC150_ROOT = Path(__file__).resolve().parents[1]
if str(_PC150_ROOT) not in sys.path:
    sys.path.insert(0, str(_PC150_ROOT))

from src.shared.checkpoints import RunPaths, save_config, save_final_metrics  # noqa: E402
from src.shared.dataset import DatasetIndex  # noqa: E402
from src.shared.device_config import detect_hardware  # noqa: E402
from src.shared.presets import resolve_preset  # noqa: E402
from src.shared.utils import get_logger, set_global_seed  # noqa: E402

_log = get_logger("train_detectron2")

VALID_MODELS = {"faster_rcnn", "retinanet", "cascade_rcnn"}

_DETECTRON2_INSTALL_HINT = """
Detectron2 não está instalado. Para instalar:

  # Com uv:
  uv pip install 'git+https://github.com/facebookresearch/detectron2.git'

  # No macOS (CPU-only):
  MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ \\
  uv pip install 'git+https://github.com/facebookresearch/detectron2.git'
"""


def _import_detectron2():
    """Importa detectron2 com mensagem clara se não instalado."""
    try:
        import detectron2  # noqa: F401
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultTrainer, HookBase
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader, MetadataCatalog
        return {
            "model_zoo": model_zoo,
            "get_cfg": get_cfg,
            "register_coco_instances": register_coco_instances,
            "DefaultTrainer": DefaultTrainer,
            "HookBase": HookBase,
            "COCOEvaluator": COCOEvaluator,
            "inference_on_dataset": inference_on_dataset,
            "build_detection_test_loader": build_detection_test_loader,
            "MetadataCatalog": MetadataCatalog,
        }
    except ImportError as exc:
        _log.error("⚠️  Detectron2 indisponível: %s", exc)
        print(_DETECTRON2_INSTALL_HINT, file=sys.stderr)
        sys.exit(2)


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treino Detectron2.")
    p.add_argument("--model", type=str, default="faster_rcnn",
                   help=f"Modelo: {sorted(VALID_MODELS)}")
    p.add_argument("--data", type=Path, help="Diretório data/processed/.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--subset", type=float, default=None)
    p.add_argument("--preset", type=str, default="balanced",
                   choices=["fast", "balanced", "full"])
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--no-augment", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
# Treino
# ──────────────────────────────────────────────────────────────────


def _load_d2_config_yaml() -> dict:
    with (_PC150_ROOT / "configs" / "detectron2_config.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _maybe_subset_coco(annotations_json: Path, subset_fraction: float, seed: int) -> Path:
    """Cria um JSON COCO reduzido (se subset < 1.0). Retorna path do arquivo a usar."""
    if subset_fraction >= 1.0:
        return annotations_json
    import random

    with annotations_json.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    ids = sorted(img["id"] for img in data["images"])
    rng = random.Random(seed)
    k = max(1, int(len(ids) * subset_fraction))
    keep = set(rng.sample(ids, k=k))
    data["images"] = [im for im in data["images"] if im["id"] in keep]
    data["annotations"] = [a for a in data["annotations"] if a["image_id"] in keep]
    out = annotations_json.with_name(annotations_json.stem + f"_subset_{int(subset_fraction*100)}.json")
    with out.open("w", encoding="utf-8") as fh:
        json.dump(data, fh)
    _log.info("Subset train Detectron2: %d/%d imagens → %s", k, len(ids), out.name)
    return out


def train(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)

    if args.model not in VALID_MODELS:
        raise SystemExit(f"Modelo inválido: {args.model}. Use: {sorted(VALID_MODELS)}")

    hw = detect_hardware()
    d2 = _import_detectron2()
    d2_cfg = _load_d2_config_yaml()

    if args.data is None and args.resume is None:
        raise SystemExit("--data é obrigatório (a menos que --resume seja usado).")

    preset = resolve_preset(
        args.preset, hw,
        epochs=args.epochs, batch=args.batch, subset=args.subset,
        no_augment=args.no_augment,
    )

    # Carrega índice e registra datasets COCO no Detectron2.
    index = DatasetIndex.from_root(args.data)
    train_json = _maybe_subset_coco(index.train.annotations_json, preset.subset, args.seed)
    val_json = index.val.annotations_json
    train_name = f"_train_{args.model}_{datetime.now().strftime('%H%M%S')}"
    val_name = f"_val_{args.model}_{datetime.now().strftime('%H%M%S')}"
    d2["register_coco_instances"](train_name, {}, str(train_json), str(index.train.images_dir))
    d2["register_coco_instances"](val_name, {}, str(val_json), str(index.val.images_dir))

    # Calcula max_iter a partir de epochs.
    with train_json.open("r", encoding="utf-8") as fh:
        n_train = len(json.load(fh)["images"])
    iters_per_epoch = max(1, n_train // max(1, preset.batch))
    max_iter = iters_per_epoch * preset.epochs
    save_every_iters = iters_per_epoch * 3  # a cada 3 épocas

    # Config base do model zoo.
    zoo_file = d2_cfg["model_zoo"][args.model]
    if args.model == "cascade_rcnn":
        # Cascade do zoo é mask; só funciona se dataset tem máscaras.
        has_masks = any("segmentation" in a and a["segmentation"] for a in
                        json.loads(train_json.read_text()).get("annotations", []))
        if not has_masks:
            raise SystemExit(
                "cascade_rcnn no Model Zoo é mask R-CNN — seu dataset não tem máscaras. "
                "Use --model faster_rcnn ou retinanet."
            )

    cfg = d2["get_cfg"]()
    cfg.merge_from_file(d2["model_zoo"].get_config_file(zoo_file))
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    cfg.DATALOADER.NUM_WORKERS = hw.num_workers
    cfg.MODEL.WEIGHTS = d2["model_zoo"].get_checkpoint_url(zoo_file)
    cfg.SOLVER.IMS_PER_BATCH = preset.batch
    cfg.SOLVER.BASE_LR = float(d2_cfg["solver"]["base_lr"])
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.WEIGHT_DECAY = float(d2_cfg["solver"]["weight_decay"])
    cfg.SOLVER.WARMUP_ITERS = int(d2_cfg["solver"]["warmup_iters"])
    cfg.SOLVER.WARMUP_FACTOR = float(d2_cfg["solver"]["warmup_factor"])
    cfg.SOLVER.MOMENTUM = float(d2_cfg["solver"]["momentum"])
    cfg.SOLVER.GAMMA = float(d2_cfg["solver"]["gamma"])
    cfg.SOLVER.CHECKPOINT_PERIOD = save_every_iters
    cfg.SOLVER.STEPS = (int(max_iter * 0.7), int(max_iter * 0.9))
    cfg.INPUT.MIN_SIZE_TRAIN = tuple(d2_cfg["input"]["min_size_train"])
    cfg.INPUT.MAX_SIZE_TRAIN = int(d2_cfg["input"]["max_size_train"])
    cfg.INPUT.MIN_SIZE_TEST = int(d2_cfg["input"]["min_size_test"])
    cfg.INPUT.MAX_SIZE_TEST = int(d2_cfg["input"]["max_size_test"])
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = index.num_classes() if hasattr(cfg.MODEL.ROI_HEADS, "NUM_CLASSES") else cfg.MODEL.ROI_HEADS.NUM_CLASSES
    try:
        cfg.MODEL.RETINANET.NUM_CLASSES = index.num_classes()
    except AttributeError:
        pass
    cfg.MODEL.DEVICE = hw.detectron2_device
    cfg.SEED = args.seed
    cfg.TEST.EVAL_PERIOD = iters_per_epoch  # avalia a cada época

    # Paths da run e OUTPUT_DIR do Detectron2.
    name = args.name or f"d2_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    paths = RunPaths.new(name)
    cfg.OUTPUT_DIR = str(paths.root)

    save_config(paths, {
        "framework": "detectron2",
        "model": args.model,
        "preset": preset.__dict__,
        "hardware": hw.__dict__,
        "args": vars(args),
        "max_iter": max_iter,
        "iters_per_epoch": iters_per_epoch,
    })

    # Hook customizado: early stopping + CSV de métricas por época.
    HookBase = d2["HookBase"]
    early_patience = int(d2_cfg["early_stopping"]["patience"])
    metric_key = d2_cfg["early_stopping"]["metric"]

    class EarlyStopAndLog(HookBase):
        def __init__(self) -> None:
            super().__init__()
            self.best = -1.0
            self.bad_epochs = 0
            self.csv_path = paths.metrics_csv
            with self.csv_path.open("w", encoding="utf-8", newline="") as fh:
                csv.writer(fh).writerow(["epoch", "iter", "loss", "mAP@0.5", "mAP@0.5:0.95"])

        def after_step(self) -> None:
            if (self.trainer.iter + 1) % iters_per_epoch != 0:
                return
            epoch = (self.trainer.iter + 1) // iters_per_epoch
            storage = self.trainer.storage
            loss = storage.history("total_loss").latest() if "total_loss" in storage._history else 0.0
            map50 = storage.histories().get("bbox/AP50")
            map5095 = storage.histories().get("bbox/AP")
            v50 = float(map50.latest()) / 100 if map50 else 0.0
            v5095 = float(map5095.latest()) / 100 if map5095 else 0.0
            with self.csv_path.open("a", encoding="utf-8", newline="") as fh:
                csv.writer(fh).writerow([epoch, self.trainer.iter + 1, float(loss), v50, v5095])

            # Best e early stopping (usa o metric configurado).
            current = v50 if "AP50" in metric_key else v5095
            ckpt_src = Path(cfg.OUTPUT_DIR) / "model_final.pth"
            last_ckpt = Path(cfg.OUTPUT_DIR) / f"model_{self.trainer.iter:07d}.pth"
            if last_ckpt.exists():
                shutil.copy2(last_ckpt, paths.last)
                # checkpoint periódico
                ckpt_target = paths.checkpoints / f"epoch_{epoch:03d}.pt"
                shutil.copy2(last_ckpt, ckpt_target)
            if current > self.best:
                self.best = current
                self.bad_epochs = 0
                if last_ckpt.exists():
                    shutil.copy2(last_ckpt, paths.best)
                _log.info("🏆 Novo best mAP=%.4f (epoch %d)", current, epoch)
            else:
                self.bad_epochs += 1
                if self.bad_epochs >= early_patience:
                    _log.warning("Early stopping em epoch %d (paciência %d esgotada)",
                                 epoch, early_patience)
                    raise StopIteration("Early stopping")

    class TrainerWithEval(d2["DefaultTrainer"]):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            return d2["COCOEvaluator"](dataset_name, output_dir=cfg.OUTPUT_DIR)

    trainer = TrainerWithEval(cfg)
    trainer.register_hooks([EarlyStopAndLog()])
    trainer.resume_or_load(resume=bool(args.resume))

    try:
        trainer.train()
    except StopIteration:
        pass  # early stopping

    # Métricas finais.
    final_pth = Path(cfg.OUTPUT_DIR) / "model_final.pth"
    if final_pth.exists() and not paths.last.exists():
        shutil.copy2(final_pth, paths.last)
    metrics_path = Path(cfg.OUTPUT_DIR) / "metrics.json"
    final = {}
    if metrics_path.exists():
        # metrics.json do Detectron2 é JSONL.
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            try:
                final.update(json.loads(line))
            except json.JSONDecodeError:
                continue
    save_final_metrics(paths, final)
    _log.info("✅ Treino Detectron2 concluído. Run: %s", paths.root)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
