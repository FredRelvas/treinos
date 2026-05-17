"""Carrega e mescla presets de treino com overrides de CLI.

Fluxo: YAML base (`configs/presets.yaml`) →
       seleciona preset + hardware →
       aplica overrides explícitos do CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from .device_config import HardwareConfig
from .utils import get_logger, project_root

_log = get_logger(__name__)

VALID_PRESETS = {"fast", "balanced", "full"}
VALID_AUG = {"off", "light", "standard", "full"}


@dataclass
class TrainingPreset:
    """Parâmetros efetivos de treino para esta combinação preset×hardware."""

    name: str
    epochs: int
    batch: int
    subset: float
    augmentation: str
    device: str  # 'cuda' | 'mps' | 'cpu'


def _presets_path() -> Path:
    return project_root() / "configs" / "presets.yaml"


def load_presets_yaml(path: Optional[Path] = None) -> dict[str, Any]:
    path = Path(path) if path else _presets_path()
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_preset(
    preset_name: str,
    hw: HardwareConfig,
    *,
    epochs: Optional[int] = None,
    batch: Optional[int] = None,
    subset: Optional[float] = None,
    no_augment: bool = False,
    yaml_path: Optional[Path] = None,
) -> TrainingPreset:
    """Resolve preset + hardware → ``TrainingPreset`` final.

    Args:
        preset_name: 'fast' | 'balanced' | 'full'.
        hw: hardware detectado.
        epochs, batch, subset: overrides do CLI (None = usa preset).
        no_augment: flag --no-augment do CLI.
        yaml_path: opcional, default = ``configs/presets.yaml``.
    """
    if preset_name not in VALID_PRESETS:
        raise ValueError(f"Preset inválido '{preset_name}'. Use: {sorted(VALID_PRESETS)}")

    data = load_presets_yaml(yaml_path)
    if preset_name not in data:
        raise KeyError(f"Preset '{preset_name}' ausente em presets.yaml")

    device_key = hw.device
    if device_key not in data[preset_name]:
        raise KeyError(
            f"Preset '{preset_name}' não define seção para device '{device_key}'"
        )

    base = data[preset_name][device_key]
    eff_epochs = int(epochs) if epochs is not None else int(base["epochs"])
    eff_batch = int(batch) if batch is not None else int(base["batch"])
    eff_subset = float(subset) if subset is not None else float(base["subset"])
    eff_aug = "off" if no_augment else str(base["augmentation"])

    if not (0.0 < eff_subset <= 1.0):
        raise ValueError(f"--subset deve estar em (0, 1]. Recebido: {eff_subset}")
    if eff_aug not in VALID_AUG:
        raise ValueError(f"Nível de augmentation inválido: '{eff_aug}'")

    out = TrainingPreset(
        name=preset_name,
        epochs=eff_epochs,
        batch=eff_batch,
        subset=eff_subset,
        augmentation=eff_aug,
        device=device_key,
    )
    _log.info(
        "Preset resolvido: %s/%s → epochs=%d batch=%d subset=%.2f aug=%s",
        preset_name,
        device_key,
        eff_epochs,
        eff_batch,
        eff_subset,
        eff_aug,
    )
    return out
