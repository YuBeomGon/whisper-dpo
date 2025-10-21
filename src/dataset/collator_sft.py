"""
Data collator for supervised Whisper fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
from transformers import WhisperProcessor

from .collator_dpo import default_audio_loader, _to_mono


@dataclass
class WhisperSFTCollator:
    processor: WhisperProcessor
    sampling_rate: int = 16000
    load_audio: Optional[Callable[[str, int], torch.Tensor]] = None
    label_pad_token_id: int = -100
    max_label_length: Optional[int] = None
    max_input_frames: Optional[int] = None
    max_input_samples: Optional[int] = None

    def __post_init__(self) -> None:
        if self.load_audio is None:
            self.load_audio = default_audio_loader
        self._feature_extractor = self.processor.feature_extractor
        self._tokenizer = self.processor.tokenizer
        if self.max_label_length is None:
            target_len = (
                getattr(self._tokenizer, "decoder_max_length", None)
                or getattr(self._tokenizer, "max_length", None)
                or getattr(self._tokenizer, "model_max_length", None)
                or getattr(self._tokenizer, "max_len_single_sentence", None)
            )
            if target_len is None:
                target_len = 448
            self.max_label_length = min(int(target_len), 448)
        else:
            self.max_label_length = min(int(self.max_label_length), 448)
        if self.max_input_frames is None:
            self.max_input_frames = getattr(self._feature_extractor, "nb_max_frames", 3000)
        if self.max_input_samples is None:
            self.max_input_samples = getattr(
                self._feature_extractor,
                "n_samples",
                self.max_input_frames * getattr(self._feature_extractor, "hop_length", 160),
            )

    def __call__(self, features: Iterable[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        feature_list = list(features)
        if not feature_list:
            raise ValueError("WhisperSFTCollator received an empty batch.")

        waveforms: List[torch.Tensor] = []
        for sample in feature_list:
            waveform = self._resolve_waveform(sample["audio_path"])
            waveforms.append(waveform)

        np_waveforms = [
            w.cpu().numpy() if isinstance(w, torch.Tensor) else np.asarray(w, dtype=np.float32)
            for w in waveforms
        ]
        feat_inputs = self._feature_extractor(
            np_waveforms,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_input_samples,
            truncation=True,
            return_attention_mask=True,
        )

        outputs = self._tokenizer(
            [sample["text"] for sample in feature_list],
            return_tensors="pt",
            padding="max_length",
            truncation=self.max_label_length is not None,
            max_length=self.max_label_length,
            return_attention_mask=True,
        )
        labels = outputs.input_ids.masked_fill(outputs.attention_mask.eq(0), self.label_pad_token_id)

        batch: Dict[str, torch.Tensor] = {
            "input_features": feat_inputs.input_features,
            "labels": labels,
        }
        if hasattr(feat_inputs, "attention_mask"):
            batch["attention_mask"] = feat_inputs.attention_mask
        return batch

    def _resolve_waveform(self, path: str) -> torch.Tensor:
        waveform = self.load_audio(path, self.sampling_rate)
        if waveform.ndim > 1:
            waveform = _to_mono(waveform)
        return waveform.float()


__all__ = ["WhisperSFTCollator"]
