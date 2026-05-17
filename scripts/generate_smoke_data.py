"""Gera/regenera o dataset sintético em ``data/smoke/``.

Cria 10 imagens PNG com formas geométricas coloridas (círculo, quadrado,
triângulo) sobre fundo aleatório e anotações COCO JSON correspondentes.
Útil para validar a pipeline sem depender do dataset real.

Executar uma vez antes da prova e commitar o conteúdo de data/smoke/.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

# Permite executar tanto como módulo quanto como script direto.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.shared.utils import ensure_dir, get_logger, set_global_seed  # noqa: E402

_log = get_logger("generate_smoke")

CLASSES = [
    {"id": 1, "name": "circle"},
    {"id": 2, "name": "square"},
    {"id": 3, "name": "triangle"},
]

IMG_W, IMG_H = 320, 320
N_IMAGES_DEFAULT = 10


def _rand_color() -> Tuple[int, int, int]:
    return (random.randint(40, 230), random.randint(40, 230), random.randint(40, 230))


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str) -> Tuple[int, int, int, int]:
    """Desenha uma forma e retorna a bbox no formato COCO [x, y, w, h]."""
    size = random.randint(40, 90)
    x = random.randint(5, IMG_W - size - 5)
    y = random.randint(5, IMG_H - size - 5)
    color = _rand_color()
    if shape == "circle":
        draw.ellipse([x, y, x + size, y + size], fill=color, outline=(0, 0, 0))
    elif shape == "square":
        draw.rectangle([x, y, x + size, y + size], fill=color, outline=(0, 0, 0))
    elif shape == "triangle":
        draw.polygon(
            [(x + size // 2, y), (x, y + size), (x + size, y + size)],
            fill=color,
            outline=(0, 0, 0),
        )
    else:
        raise ValueError(f"Forma desconhecida: {shape}")
    return (x, y, size, size)


def _gen_image(idx: int, out_dir: Path) -> Tuple[dict, list[dict]]:
    """Gera uma imagem e retorna (image_record, [annotation_records])."""
    bg = np.full((IMG_H, IMG_W, 3), random.randint(200, 255), dtype=np.uint8)
    bg += np.random.randint(-15, 15, bg.shape, dtype=np.int16).astype(np.uint8)
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    annotations = []
    n_objs = random.randint(2, 4)
    next_ann_id = idx * 100  # garante IDs únicos sem colisão
    for j in range(n_objs):
        cls = random.choice(CLASSES)
        bx, by, bw, bh = _draw_shape(draw, cls["name"])
        annotations.append(
            {
                "id": next_ann_id + j,
                "image_id": idx,
                "category_id": cls["id"],
                "bbox": [bx, by, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "segmentation": [],
            }
        )

    file_name = f"smoke_{idx:03d}.png"
    img.save(out_dir / file_name)
    image_record = {
        "id": idx,
        "file_name": file_name,
        "width": IMG_W,
        "height": IMG_H,
    }
    return image_record, annotations


def generate(out_root: Path, n_images: int = N_IMAGES_DEFAULT, seed: int = 42) -> Path:
    """Gera o dataset sintético completo. Retorna o path do JSON de anotações."""
    set_global_seed(seed)
    img_dir = ensure_dir(out_root / "images")

    # Limpa imagens antigas para garantir consistência com o JSON.
    for old in img_dir.glob("smoke_*.png"):
        old.unlink()

    images, anns = [], []
    for i in range(1, n_images + 1):
        rec, a = _gen_image(i, img_dir)
        images.append(rec)
        anns.extend(a)

    coco = {
        "info": {"description": "Smoke dataset sintético (formas geométricas)"},
        "licenses": [],
        "images": images,
        "annotations": anns,
        "categories": CLASSES,
    }
    json_path = out_root / "annotations.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(coco, fh, indent=2)

    _log.info(
        "Geradas %d imagens e %d anotações em %s", len(images), len(anns), out_root
    )
    return json_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gera dataset sintético em data/smoke/")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "smoke",
        help="Diretório de saída (padrão: PC-150/data/smoke)",
    )
    p.add_argument("--n", type=int, default=N_IMAGES_DEFAULT, help="Número de imagens")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate(args.out, n_images=args.n, seed=args.seed)
    readme = args.out / "README.md"
    if not readme.exists():
        readme.write_text(
            "# data/smoke/\n\n"
            "Dataset sintético usado pelo `smoke_test.py --mode quick`.\n\n"
            "Para regenerar:\n\n"
            "```bash\n"
            "python scripts/generate_smoke_data.py\n"
            "```\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
