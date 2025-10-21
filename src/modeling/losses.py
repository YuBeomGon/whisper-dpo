"""
Auxiliary loss utilities for Whisper preference optimization.
"""

from __future__ import annotations

import torch
from torch import Tensor


def calibration_anchor(preferred_logps_policy: Tensor, preferred_logps_reference: Tensor) -> Tensor:
    """
    Compute Cal-DPO style anchor penalty to discourage the policy from lowering
    the absolute likelihood of preferred sequences relative to the reference.

    Parameters
    ----------
    preferred_logps_policy:
        Log-probabilities of preferred responses under the policy.
    preferred_logps_reference:
        Log-probabilities of preferred responses under the reference model.
    """
    if preferred_logps_policy.shape != preferred_logps_reference.shape:
        raise ValueError("Policy and reference log-probabilities must share shape.")
    return torch.relu(preferred_logps_reference - preferred_logps_policy)


__all__ = ["calibration_anchor"]

