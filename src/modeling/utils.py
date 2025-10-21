"""
Utility helpers shared across Whisper modeling components.
"""

from __future__ import annotations

import copy
from typing import Optional

from transformers import PreTrainedModel


def freeze_backbone(model: PreTrainedModel) -> None:
    """Disable gradients for all parameters in-place."""
    for param in model.parameters():
        param.requires_grad_(False)


def clone_reference_model(model: PreTrainedModel) -> PreTrainedModel:
    """
    Create a reference copy of `model` with gradients disabled.

    LoRA/PEFT layers are preserved because the copy is created via deepcopy.
    """
    reference = copy.deepcopy(model)
    freeze_backbone(reference)
    return reference


def maybe_unfreeze_lora_peft(model: PreTrainedModel) -> None:
    """
    Re-enable gradients for adapter parameters that advertise `requires_grad`.

    Some PEFT modules manage their own parameter flags; this helper ensures
    the flag is set to True when the base backbone is frozen.
    """
    for name, param in model.named_parameters():
        if "lora" in name.lower() or "adapter" in name.lower():
            param.requires_grad_(True)


__all__ = ["freeze_backbone", "clone_reference_model", "maybe_unfreeze_lora_peft"]

