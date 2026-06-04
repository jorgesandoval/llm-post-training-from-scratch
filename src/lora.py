"""Manual LoRA (Low-Rank Adaptation) implemented from scratch in pure PyTorch.

This module deliberately does NOT use the `peft` library. The whole point is to
show the mechanics: a LoRA layer is just a frozen base linear plus a trainable
low-rank residual ``(alpha / r) * B @ A``.

Reference: Hu et al., 2021, "LoRA: Low-Rank Adaptation of Large Language Models"
(https://arxiv.org/abs/2106.09685).
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A drop-in replacement for ``nn.Linear`` with a trainable low-rank update.

    The forward pass computes::

        y = x W^T + b            (frozen base layer, never updated)
          + (alpha / r) * (x A^T) B^T   (trainable low-rank residual)

    where ``A`` has shape ``(r, in_features)`` and ``B`` has shape
    ``(out_features, r)``. ``B`` is initialized to zero so that at the start of
    training the module is mathematically identical to the original linear layer.

    Args:
        base_linear: The pretrained ``nn.Linear`` to wrap. Its weights are frozen.
        r: LoRA rank (the bottleneck dimension). Higher r => more capacity.
        alpha: LoRA scaling factor. The effective scale is ``alpha / r``.
        dropout: Dropout applied to the input of the LoRA branch only.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {r}.")

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # HERE: the original weights are kept verbatim and FROZEN. LoRA never
        # touches them; it only learns an additive low-rank correction.
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # Low-rank trainable factors. A: (r, in), B: (out, r).
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        # A ~ Kaiming uniform (like a normal linear), B = 0 so B@A = 0 initially.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        # Low-rank path: project down to r, then back up to out_features.
        lora_out = self.lora_dropout(x) @ self.lora_A.t()  # (..., r)
        lora_out = lora_out @ self.lora_B.t()              # (..., out_features)
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4f}"
        )


def _get_submodule(model: nn.Module, dotted_name: str) -> nn.Module:
    """Return the submodule referenced by a dotted path (e.g. ``a.b.0.c``)."""
    module = model
    for part in dotted_name.split("."):
        module = getattr(module, part)
    return module


def _set_submodule(model: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Replace the submodule referenced by a dotted path with ``new_module``."""
    parent_path, _, child_name = dotted_name.rpartition(".")
    parent = _get_submodule(model, parent_path) if parent_path else model
    setattr(parent, child_name, new_module)


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str] = ("q_proj", "v_proj"),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    freeze_base: bool = True,
) -> int:
    """Replace every matching ``nn.Linear`` in ``model`` with a ``LoRALinear``.

    Args:
        model: The model to modify in place (e.g. an ``AutoModelForCausalLM``).
        target_modules: Substrings matched against module names. By convention we
            adapt the attention query/value projections (``q_proj``, ``v_proj``).
        r, alpha, dropout: LoRA hyper-parameters forwarded to ``LoRALinear``.
        freeze_base: If True, first freeze ALL parameters so that only the newly
            created LoRA factors remain trainable.

    Returns:
        The number of linear layers that were wrapped.
    """
    if freeze_base:
        for param in model.parameters():
            param.requires_grad_(False)

    # Collect targets first; we cannot mutate the module tree while iterating it.
    to_replace: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            to_replace.append((name, module))

    for name, linear in to_replace:
        lora_layer = LoRALinear(linear, r=r, alpha=alpha, dropout=dropout)
        _set_submodule(model, name, lora_layer)

    return len(to_replace)


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return only the trainable LoRA parameters (for the optimizer)."""
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract just the LoRA weights so checkpoints stay tiny (a few MB)."""
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load LoRA weights produced by :func:`lora_state_dict` back into a model."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key, b_key = f"{name}.lora_A", f"{name}.lora_B"
            if a_key in state and b_key in state:
                with torch.no_grad():
                    module.lora_A.copy_(state[a_key].to(module.lora_A.device))
                    module.lora_B.copy_(state[b_key].to(module.lora_B.device))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
