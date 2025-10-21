"""
Custom TRL trainer that extends `DPOTrainer` to support Whisper-style
encoder-decoder models trained on audio preference data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel
from trl import DPOTrainer
from datasets import Dataset, IterableDataset

from .losses import calibration_anchor


def _mask_pad_tokens(labels: Tensor, pad_token_id: int = -100) -> Tuple[Tensor, Tensor]:
    """
    Build a boolean mask for valid tokens and a version of labels where masked
    positions are safe to gather from.
    """
    mask = labels.ne(pad_token_id)
    safe_labels = torch.where(mask, labels, torch.zeros_like(labels))
    return mask, safe_labels


def sequence_logprobs(
    model: PreTrainedModel,
    input_features: Tensor,
    labels: Tensor,
    attention_mask: Optional[Tensor] = None,
    length_strategy: str = "average",
) -> Tensor:
    """
    Compute log-probabilities of target sequences under a Whisper policy.

    Parameters
    ----------
    model:
        Whisper encoder-decoder model (policy or reference).
    input_features:
        Log-mel features as produced by `WhisperProcessor.feature_extractor`.
    labels:
        Tokenized targets with padding masked by -100.
    attention_mask:
        Optional encoder attention mask.
    length_strategy:
        How to normalize across token positions. Options: {"average", "sum"}.
    """
    outputs = model(
        input_features=input_features,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits  # [B, T, V]
    log_probs = logits.log_softmax(dim=-1)
    mask, safe_labels = _mask_pad_tokens(labels)
    token_log_probs = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * mask
    lengths = mask.sum(dim=-1).clamp(min=1)
    summed = token_log_probs.sum(dim=-1)
    if length_strategy == "average":
        return summed / lengths
    if length_strategy == "sum":
        return summed
    raise ValueError(f"Unsupported length normalization strategy: {length_strategy}")


@dataclass
class WhisperLossOutputs:
    pos_logps_policy: Tensor
    neg_logps_policy: Tensor
    pos_logps_reference: Tensor
    neg_logps_reference: Tensor


class WhisperDPOTrainer(DPOTrainer):
    """
    DPO trainer specialized for Whisper encoder-decoder models.

    Adds support for audio features (`input_features`) and injectable loss
    shaping such as length normalization and Cal-style anchoring.
    """

    def __init__(
        self,
        *args,
        length_normalization: str = "average",
        calibration_loss_weight: float = 0.0,
        **kwargs,
    ):
        self.length_normalization = length_normalization
        self.calibration_loss_weight = calibration_loss_weight
        super().__init__(*args, **kwargs)

    def _length_strategy(self) -> str:
        return self.length_normalization

    def _calibration_weight(self) -> float:
        return float(self.calibration_loss_weight)

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class,
        args,
        dataset_name: str,
    ):
        # Skip TRL's text-specific preprocessing; Whisper collator handles raw fields.
        return dataset

    def concatenated_forward(
        self,
        model: PreTrainedModel,
        batch: Dict[str, Tensor],
        is_ref_model: bool = False,
    ) -> Dict[str, Tensor]:
        """Compute log-prob outputs for eval/reference paths using Whisper inputs."""
        input_features = batch["input_features"]
        attention_mask = batch.get("attention_mask")
        positive_labels = batch["labels_pos"]
        negative_labels = batch["labels_neg"]

        chosen_logps = sequence_logprobs(
            model,
            input_features=input_features,
            labels=positive_labels,
            attention_mask=attention_mask,
            length_strategy=self._length_strategy(),
        )
        rejected_logps = sequence_logprobs(
            model,
            input_features=input_features,
            labels=negative_labels,
            attention_mask=attention_mask,
            length_strategy=self._length_strategy(),
        )

        output: Dict[str, Tensor] = {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "mean_chosen_logits": chosen_logps.mean(),
            "mean_rejected_logits": rejected_logps.mean(),
        }
        return output

    def compute_loss(  # type: ignore[override]
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        input_features = inputs["input_features"]
        attention_mask = inputs.get("attention_mask", None)
        positive_labels = inputs["labels_pos"]
        negative_labels = inputs["labels_neg"]

        # Policy log-probs require gradients.
        pos_logps_pi = sequence_logprobs(
            model,
            input_features=input_features,
            labels=positive_labels,
            attention_mask=attention_mask,
            length_strategy=self._length_strategy(),
        )
        neg_logps_pi = sequence_logprobs(
            model,
            input_features=input_features,
            labels=negative_labels,
            attention_mask=attention_mask,
            length_strategy=self._length_strategy(),
        )

        # Reference log-probs are detached.
        with torch.no_grad():
            ref_model = self.ref_model if hasattr(self, "ref_model") else None
            if ref_model is None:
                raise RuntimeError(
                    "WhisperDPOTrainer requires a reference model. "
                    "Ensure `ref_model` is provided during initialization."
                )
            pos_logps_ref = sequence_logprobs(
                ref_model,
                input_features=input_features,
                labels=positive_labels,
                attention_mask=attention_mask,
                length_strategy=self._length_strategy(),
            )
            neg_logps_ref = sequence_logprobs(
                ref_model,
                input_features=input_features,
                labels=negative_labels,
                attention_mask=attention_mask,
                length_strategy=self._length_strategy(),
            )

        # Standard DPO loss.
        beta = getattr(self, "beta", getattr(self.args, "beta", 0.1))
        policy_pref_diff = pos_logps_pi - neg_logps_pi
        reference_pref_diff = pos_logps_ref - neg_logps_ref
        logits = beta * (policy_pref_diff - reference_pref_diff)
        loss = -F.logsigmoid(logits).mean()

        # Optional Cal-style anchor for absolute preference likelihood.
        cal_weight = self._calibration_weight()
        if cal_weight > 0.0:
            anchor = calibration_anchor(pos_logps_pi, pos_logps_ref).mean()
            loss = loss + cal_weight * anchor

        if return_outputs:
            aux = WhisperLossOutputs(
                pos_logps_policy=pos_logps_pi.detach(),
                neg_logps_policy=neg_logps_pi.detach(),
                pos_logps_reference=pos_logps_ref,
                neg_logps_reference=neg_logps_ref,
            )
            return loss, aux
        return loss


__all__ = ["WhisperDPOTrainer", "sequence_logprobs", "WhisperLossOutputs"]
