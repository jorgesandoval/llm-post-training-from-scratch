"""Shared utilities: reproducibility, device selection, logging, checkpointing, plots.

These helpers are intentionally small and dependency-light so that each notebook
can focus on the *algorithm* rather than on boilerplate.
"""

from __future__ import annotations

import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ContextManager

import numpy as np
import torch

from .lora import lora_state_dict, load_lora_state_dict


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs.

    Args:
        seed: The random seed.
        deterministic: If True, request deterministic cuDNN kernels. This can
            slow things down slightly but makes results repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available device (CUDA > Apple MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device) -> torch.dtype:
    """Pick a sensible mixed-precision dtype for ``torch.autocast``.

    bf16 on Ampere+ GPUs (more numerically robust), fp16 elsewhere on CUDA,
    fp32 on CPU/MPS.
    """
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def amp_context(device: torch.device) -> ContextManager:
    """Return a mixed-precision autocast context that is safe on any device.

    * CUDA with bf16 support  -> ``torch.autocast`` in bfloat16 (no GradScaler
      needed, which keeps the training loops simple and didactic).
    * Everything else (MPS / CPU / old CUDA) -> a no-op context (full fp32).

    This lets the exact same training loop run on a MacBook (Metal) and on a
    RunPod GPU without changes.
    """
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@dataclass
class LossLogger:
    """Tiny in-memory logger for scalar training metrics.

    Example:
        logger = LossLogger()
        logger.log(step=10, loss=2.31)
        logger.log(step=20, loss=2.05, reward=0.4)
    """

    history: dict[str, list[tuple[int, float]]] = field(default_factory=dict)

    def log(self, step: int, **metrics: float) -> None:
        for name, value in metrics.items():
            self.history.setdefault(name, []).append((step, float(value)))

    def series(self, name: str) -> tuple[list[int], list[float]]:
        points = self.history.get(name, [])
        steps = [s for s, _ in points]
        values = [v for _, v in points]
        return steps, values

    def last(self, name: str) -> float | None:
        points = self.history.get(name)
        return points[-1][1] if points else None


def plot_curves(
    logger: LossLogger,
    metrics: list[str] | None = None,
    title: str = "Training curves",
    ylabel: str = "value",
    smooth: int = 1,
):
    """Plot one or more metric curves from a :class:`LossLogger`.

    Returns the matplotlib ``Figure`` so notebooks can further customize it.
    """
    import matplotlib.pyplot as plt

    metrics = metrics or list(logger.history.keys())
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in metrics:
        steps, values = logger.series(name)
        if not values:
            continue
        if smooth > 1 and len(values) >= smooth:
            kernel = np.ones(smooth) / smooth
            values = np.convolve(values, kernel, mode="valid").tolist()
            steps = steps[smooth - 1:]
        ax.plot(steps, values, label=name)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def save_lora_checkpoint(
    model: torch.nn.Module,
    path: str | os.PathLike,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save ONLY the LoRA weights (plus optional metadata) to ``path``.

    Keeping checkpoints LoRA-only means each stage is just a few megabytes and the
    chained pipeline (CPT -> SFT -> DPO -> GRPO) stays cheap.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"lora": lora_state_dict(model), "metadata": metadata or {}}
    torch.save(payload, path)
    return path


def load_lora_checkpoint(
    model: torch.nn.Module,
    path: str | os.PathLike,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load LoRA weights saved by :func:`save_lora_checkpoint` into ``model``.

    Returns the stored metadata dict.
    """
    payload = torch.load(path, map_location=map_location)
    load_lora_state_dict(model, payload["lora"])
    return payload.get("metadata", {})


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    """Read a JSON Lines file into a list of dicts."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | os.PathLike, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts to a JSON Lines file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
