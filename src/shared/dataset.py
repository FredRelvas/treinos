"""Carregamento de datasets COCO JSON e utilitários de split/subset.

Fornece:
  - ``CocoDetectionDataset``: ``torch.utils.data.Dataset`` minimalista que
    devolve ``(image_tensor, target_dict)`` no padrão torchvision.
  - ``DatasetIndex``: descrição de splits (train/val/test) localizados em disco.
  - Funções para detectar formato de dataset e aplicar subset reprodutível.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .utils import get_logger

_log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────
# Índice de splits em disco
# ──────────────────────────────────────────────────────────────────


@dataclass
class SplitPaths:
    """Localização dos arquivos COCO de um split."""

    images_dir: Path
    annotations_json: Path

    def exists(self) -> bool:
        return self.images_dir.is_dir() and self.annotations_json.is_file()


@dataclass
class DatasetIndex:
    """Índice de um dataset COCO já processado em ``data/processed/``.

    Layout esperado:
        root/
          train/images/  train/annotations.json
          val/images/    val/annotations.json
          test/images/   test/annotations.json   (opcional)
    """

    root: Path
    train: SplitPaths
    val: SplitPaths
    test: Optional[SplitPaths] = None
    categories: list[dict] = field(default_factory=list)

    @classmethod
    def from_root(cls, root: Path) -> "DatasetIndex":
        root = Path(root)
        train = SplitPaths(root / "train" / "images", root / "train" / "annotations.json")
        val = SplitPaths(root / "val" / "images", root / "val" / "annotations.json")
        test = SplitPaths(root / "test" / "images", root / "test" / "annotations.json")
        if not train.exists():
            raise FileNotFoundError(
                f"Split de treino não encontrado em {train.images_dir} / {train.annotations_json}"
            )
        if not val.exists():
            raise FileNotFoundError(
                f"Split de validação não encontrado em {val.images_dir}. "
                "Rode `prepare_dataset.py` antes."
            )
        with train.annotations_json.open("r", encoding="utf-8") as fh:
            cats = json.load(fh).get("categories", [])
        return cls(root=root, train=train, val=val, test=test if test.exists() else None, categories=cats)

    def num_classes(self) -> int:
        return len(self.categories)

    def class_names(self) -> list[str]:
        return [c["name"] for c in sorted(self.categories, key=lambda c: c["id"])]


# ──────────────────────────────────────────────────────────────────
# Dataset torch
# ──────────────────────────────────────────────────────────────────


class CocoDetectionDataset:
    """Dataset COCO minimalista (sem dependência de torchvision.datasets.CocoDetection).

    Retorna ``(image, target)`` onde:
      - ``image`` é um ``torch.FloatTensor`` (C, H, W) normalizado em [0, 1].
      - ``target`` é um dict com ``boxes`` (N, 4) em XYXY absoluto,
        ``labels`` (N,), ``image_id`` (scalar), ``area`` (N,), ``iscrowd`` (N,).
    """

    def __init__(
        self,
        images_dir: Path,
        annotations_json: Path,
        *,
        transforms=None,
        subset_fraction: float = 1.0,
        seed: int = 42,
    ) -> None:
        import torch  # import tardio para não exigir torch em scripts utilitários

        self._torch = torch
        self.images_dir = Path(images_dir)
        with Path(annotations_json).open("r", encoding="utf-8") as fh:
            coco = json.load(fh)

        self.images = {img["id"]: img for img in coco["images"]}
        self.categories = coco.get("categories", [])
        self._cat_id_to_contiguous = {
            c["id"]: i for i, c in enumerate(sorted(self.categories, key=lambda c: c["id"]))
        }

        # Agrupa anotações por image_id.
        self.anns_by_image: dict[int, list[dict]] = {img_id: [] for img_id in self.images}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

        ids = sorted(self.images.keys())
        if not (0.0 < subset_fraction <= 1.0):
            raise ValueError(f"subset_fraction inválido: {subset_fraction}")
        if subset_fraction < 1.0:
            rng = random.Random(seed)
            k = max(1, int(len(ids) * subset_fraction))
            ids = sorted(rng.sample(ids, k=k))
            _log.info("Subset aplicado: %d/%d imagens", k, len(self.images))
        self.ids = ids
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        torch = self._torch
        image_id = self.ids[idx]
        info = self.images[image_id]
        path = self.images_dir / info["file_name"]
        img = Image.open(path).convert("RGB")

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in self.anns_by_image.get(image_id, []):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self._cat_id_to_contiguous[ann["category_id"]])
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            areas_t = torch.as_tensor(areas, dtype=torch.float32)
            crowd_t = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)
            crowd_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": areas_t,
            "iscrowd": crowd_t,
        }

        img_array = np.asarray(img, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(img_array).permute(2, 0, 1).contiguous()

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target


def collate_fn(batch):
    """Collate para detection: mantém listas (imagens podem ter shapes diferentes)."""
    images, targets = zip(*batch)
    return list(images), list(targets)
