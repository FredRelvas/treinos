"""Smoke test da pipeline em dois modos.

--mode quick : valida o ambiente em ~30s usando data/smoke/ (CPU forçado).
--mode full  : valida o dataset real (--data) com 10% e 3 épocas.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

_PC150_ROOT = Path(__file__).resolve().parents[1]
if str(_PC150_ROOT) not in sys.path:
    sys.path.insert(0, str(_PC150_ROOT))

from src.shared.utils import ensure_dir, get_logger, set_global_seed  # noqa: E402

_log = get_logger("smoke_test")


# ──────────────────────────────────────────────────────────────────
# Modo QUICK
# ──────────────────────────────────────────────────────────────────


def quick_mode() -> int:
    """Roda checks rápidos sem precisar de GPU nem dataset real."""
    print("─" * 60)
    print("SMOKE TEST — modo QUICK")
    print("─" * 60)

    smoke_dir = _PC150_ROOT / "data" / "smoke"
    ann = smoke_dir / "annotations.json"
    if not ann.exists():
        print(f"❌ {ann} não existe. Rode: python scripts/generate_smoke_data.py")
        return 1

    try:
        # 1. Imports principais.
        print("[1/6] Importando módulos compartilhados...", end=" ", flush=True)
        from src.shared.device_config import detect_hardware
        from src.shared.dataset import CocoDetectionDataset, collate_fn
        from src.shared.augmentation import build_pipeline
        from src.shared.checkpoints import RunPaths, save_config, save_final_metrics
        from src.shared.presets import resolve_preset
        print("OK")

        # 2. Detecção de hardware (não força nada, apenas verifica que roda).
        print("[2/6] Detectando hardware...", end=" ", flush=True)
        hw = detect_hardware()
        print(f"OK ({hw.device})")

        # 3. Resolução de preset.
        print("[3/6] Resolvendo preset fast...", end=" ", flush=True)
        preset = resolve_preset("fast", hw, epochs=1, batch=2)
        print(f"OK (epochs={preset.epochs}, batch={preset.batch})")

        # 4. Dataset loader + transforms.
        print("[4/6] Carregando dataset sintético...", end=" ", flush=True)
        ds = CocoDetectionDataset(
            smoke_dir / "images", ann,
            transforms=build_pipeline("light"),
        )
        assert len(ds) == 10, f"esperado 10 imagens, obtido {len(ds)}"
        img, target = ds[0]
        assert img.ndim == 3 and img.shape[0] == 3, f"shape inesperado: {img.shape}"
        assert "boxes" in target and "labels" in target
        print(f"OK ({len(ds)} imagens, primeira: {tuple(img.shape)})")

        # 5. Um forward pass com modelo torchvision leve em CPU.
        print("[5/6] Forward pass com fasterrcnn_resnet50_fpn (CPU)...", end=" ", flush=True)
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn

        torch.set_num_threads(2)
        set_global_seed(42)
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=4).eval()
        with torch.no_grad():
            out = model([img])
        assert isinstance(out, list) and isinstance(out[0], dict)
        print("OK")

        # 6. Cria run, salva config + métricas dummy.
        print("[6/6] Criando run + salvando config/métricas...", end=" ", flush=True)
        paths = RunPaths.new("smoke_quick")
        save_config(paths, {"smoke": True, "device": hw.device})
        save_final_metrics(paths, {"mAP@0.5": 0.0, "smoke_test": True})
        assert paths.config.exists() and paths.metrics.exists()
        # Limpa a run de smoke para não poluir runs/.
        shutil.rmtree(paths.root, ignore_errors=True)
        print("OK")

        print("\n✅ Ambiente OK — pipeline pronta para uso")
        return 0
    except Exception:
        print("\n❌ FALHA no smoke test:")
        traceback.print_exc()
        return 1


# ──────────────────────────────────────────────────────────────────
# Modo FULL
# ──────────────────────────────────────────────────────────────────


def full_mode(data: Path) -> int:
    """Valida o dataset real e roda 3 épocas com 10% dos dados."""
    print("─" * 60)
    print(f"SMOKE TEST — modo FULL ({data})")
    print("─" * 60)
    try:
        from src.shared.dataset import DatasetIndex
        from src.shared.device_config import detect_hardware

        hw = detect_hardware()
        index = DatasetIndex.from_root(data)
        print(f"  Classes ({index.num_classes()}): {index.class_names()}")
        print(f"  Train: {index.train.images_dir}")
        print(f"  Val:   {index.val.images_dir}")
        print(f"  Test:  {index.test.images_dir if index.test else 'n/a'}")

        print("\n→ Treinando YOLO yolov8n por 3 épocas (subset=0.1)...")
        from src import train_yolo

        args = argparse.Namespace(
            model="yolov8n",
            data=data,
            epochs=3,
            batch=None,
            subset=0.1,
            preset="fast",
            resume=None,
            seed=42,
            name="smoke_full",
            no_augment=False,
        )
        train_yolo.train(args)
        print("\n✅ Smoke FULL concluído — métricas e plots em runs/exp_*_smoke_full/")
        return 0
    except SystemExit as exc:
        print(f"\n❌ Smoke FULL parou: {exc}")
        return int(exc.code) if isinstance(exc.code, int) else 1
    except Exception:
        print("\n❌ FALHA no smoke FULL:")
        traceback.print_exc()
        return 1


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test da pipeline.")
    p.add_argument("--mode", choices=["quick", "full"], default="quick")
    p.add_argument("--data", type=Path, default=None, help="Dataset processed/ (modo full).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "quick":
        return quick_mode()
    if args.data is None:
        raise SystemExit("--data é obrigatório no --mode full.")
    return full_mode(args.data)


if __name__ == "__main__":
    sys.exit(main())
