"""Detecção automática de hardware e configuração de runtime.

Define um ``HardwareConfig`` que descreve o ambiente atual (CUDA, MPS, CPU)
com defaults sensatos para cada cenário. Outras partes da pipeline consomem
esta config para escolher dtype, batch size, num_workers, etc.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Literal

from .utils import get_logger

DeviceType = Literal["cuda", "mps", "cpu"]
_log = get_logger(__name__)


@dataclass(frozen=True)
class HardwareConfig:
    """Snapshot imutável das capacidades do hardware atual."""

    device: DeviceType
    dtype: str  # "float16" | "float32"
    use_amp: bool
    batch_size: int  # default — sobrescrito pelo preset
    num_workers: int
    detectron2_device: DeviceType  # pode diferir do device principal (MPS→CPU)
    detectron2_supported: bool
    description: str = ""
    extras: dict = field(default_factory=dict)


def _detect_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _detect_mps() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        return False


def detect_hardware() -> HardwareConfig:
    """Retorna ``HardwareConfig`` para o ambiente atual.

    Ordem de detecção: cuda → mps → cpu.
    """
    if _detect_cuda():
        import torch

        gpu_name = torch.cuda.get_device_name(0)
        cfg = HardwareConfig(
            device="cuda",
            dtype="float16",
            use_amp=True,
            batch_size=16,
            num_workers=8,
            detectron2_device="cuda",
            detectron2_supported=True,
            description=f"CUDA — {gpu_name}",
            extras={"cuda_version": torch.version.cuda or "unknown"},
        )
        _log.info("🟢 Hardware detectado: %s", cfg.description)
        return cfg

    if _detect_mps():
        cfg = HardwareConfig(
            device="mps",
            dtype="float32",
            use_amp=False,
            batch_size=8,
            num_workers=0,  # OBRIGATÓRIO no macOS (multiprocessing trava)
            detectron2_device="cpu",  # Detectron2 não suporta MPS
            detectron2_supported=True,
            description=f"MPS — Apple Silicon ({platform.machine()})",
            extras={"platform": platform.platform()},
        )
        _log.info("🟡 Hardware detectado: %s", cfg.description)
        _log.warning(
            "⚠️  Detectron2 não suporta MPS. Será forçado para CPU — espere lentidão."
        )
        return cfg

    cfg = HardwareConfig(
        device="cpu",
        dtype="float32",
        use_amp=False,
        batch_size=2,
        num_workers=0,
        detectron2_device="cpu",
        detectron2_supported=True,
        description=f"CPU — {platform.processor() or platform.machine()}",
    )
    _log.warning(
        "⚠️  Nenhuma GPU detectada. Reduza --epochs e use --preset fast."
    )
    return cfg


def device_kind(cfg: HardwareConfig) -> str:
    """Chave usada em ``configs/presets.yaml`` ('cuda' | 'mps' | 'cpu')."""
    return cfg.device
