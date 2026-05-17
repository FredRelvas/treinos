"""Pipeline de data augmentation para detecção.

Implementa transforms simples que operam em (image_tensor, target_dict),
preservando bboxes corretamente. Compatível com o ``CocoDetectionDataset``.

Para YOLO/ultralytics, augmentation é controlada via parâmetros do
``model.train()`` — este módulo cobre os pipelines genéricos (Detectron2
custom, debug, eval samples).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional


# ──────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────


@dataclass
class RandomHorizontalFlip:
    """Flip horizontal com probabilidade ``p``. Ajusta bboxes em XYXY."""

    p: float = 0.5

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        import torch

        _, _, w = image.shape  # (C, H, W)
        image = torch.flip(image, dims=[2])
        boxes = target["boxes"].clone()
        if boxes.numel():
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = w - x2
            boxes[:, 2] = w - x1
            target["boxes"] = boxes
        return image, target


@dataclass
class ColorJitter:
    """Jitter de brilho/contraste/saturação aplicado no tensor."""

    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2

    def __call__(self, image, target):
        import torch

        # Brightness
        if self.brightness > 0:
            factor = 1.0 + (random.random() * 2 - 1) * self.brightness
            image = (image * factor).clamp(0, 1)
        # Contrast (em torno da média)
        if self.contrast > 0:
            factor = 1.0 + (random.random() * 2 - 1) * self.contrast
            mean = image.mean()
            image = ((image - mean) * factor + mean).clamp(0, 1)
        # Saturation (em torno do grayscale)
        if self.saturation > 0:
            factor = 1.0 + (random.random() * 2 - 1) * self.saturation
            gray = image.mean(dim=0, keepdim=True)
            image = (gray + (image - gray) * factor).clamp(0, 1)
        return image, target


@dataclass
class RandomScale:
    """Re-escala isotrópica aleatória em [min_scale, max_scale] e ajusta bboxes."""

    min_scale: float = 0.8
    max_scale: float = 1.2

    def __call__(self, image, target):
        import torch
        import torch.nn.functional as F

        scale = random.uniform(self.min_scale, self.max_scale)
        _, h, w = image.shape
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        image = F.interpolate(
            image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
        ).squeeze(0)
        if target["boxes"].numel():
            target["boxes"] = target["boxes"] * scale
        if "area" in target and target["area"].numel():
            target["area"] = target["area"] * (scale ** 2)
        return image, target


@dataclass
class Compose:
    """Composição sequencial de transforms."""

    transforms: list

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# ──────────────────────────────────────────────────────────────────
# Pipelines pré-configurados por nível
# ──────────────────────────────────────────────────────────────────


def build_pipeline(level: str) -> Optional[Callable]:
    """Constrói um pipeline a partir do nome do nível.

    Args:
        level: "off" | "light" | "standard" | "full".

    Returns:
        Callable que recebe ``(image, target)`` ou ``None`` se ``level == "off"``.
    """
    if level == "off":
        return None
    if level == "light":
        return Compose([RandomHorizontalFlip(p=0.5), ColorJitter(0.1, 0.1, 0.1)])
    if level == "standard":
        return Compose(
            [
                RandomHorizontalFlip(p=0.5),
                ColorJitter(0.2, 0.2, 0.2),
                RandomScale(0.85, 1.15),
            ]
        )
    if level == "full":
        # Mosaic/MixUp ficam no ultralytics (parâmetros do model.train()).
        return Compose(
            [
                RandomHorizontalFlip(p=0.5),
                ColorJitter(0.3, 0.3, 0.3),
                RandomScale(0.7, 1.3),
            ]
        )
    raise ValueError(f"Nível de augmentation desconhecido: '{level}'")


def yolo_augmentation_kwargs(level: str, yolo_overrides: dict) -> dict:
    """Retorna kwargs para ``model.train()`` do ultralytics conforme o nível.

    Args:
        level: nome do nível.
        yolo_overrides: bloco ``augmentation_overrides`` do ``yolo_config.yaml``.
    """
    if level == "off":
        return {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "fliplr": 0.0,
            "flipud": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "translate": 0.0,
            "scale": 0.0,
        }
    if level not in yolo_overrides:
        raise KeyError(f"yolo_config.yaml não contém augmentation_overrides['{level}']")
    return dict(yolo_overrides[level])
