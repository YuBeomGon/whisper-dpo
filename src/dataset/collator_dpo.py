"""
Batch collation utilities for Whisper-based DPO training.

This module converts raw audio + text preference samples into the tensors that
`WhisperDPOTrainer` expects. It relies on a `WhisperProcessor` instance to
normalize audio into log-mel features and to tokenize positive/negative text
responses. The collator is agnostic to how samples are loaded; it accepts raw
numpy arrays, torchaudio tensors, or simple file paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
from transformers import WhisperProcessor

try:
    import torchaudio
except ImportError:  # pragma: no cover - torchaudio is optional at runtime
    torchaudio = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - fallback for audio loading
    sf = None

try:
    import librosa
except ImportError:  # pragma: no cover - used for resampling fallback
    librosa = None

ArrayLike = Any
Sample = Dict[str, Any]
Batch = Dict[str, torch.Tensor]


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Ensure waveform is mono by averaging across the channel dimension if needed."""
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim == 2:
        # torchaudio uses [channels, time]; average channels for mono.
        return waveform.mean(dim=0)
    raise ValueError(f"Unsupported waveform shape for mono conversion: {waveform.shape}")


def default_audio_loader(path: str, sampling_rate: int) -> torch.Tensor:
    """
    Load audio from disk using torchaudio if available, otherwise fall back to soundfile.

    Parameters
    ----------
    path:
        Location of the audio file.
    sampling_rate:
        Target sampling rate.  Audio will be resampled on the fly if needed.
    """
    if torchaudio is not None:
        waveform, sr = torchaudio.load(path)  # [channels, time]
        waveform = _to_mono(waveform)
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
            waveform = resampler(waveform)
        return waveform

    if sf is None:
        raise RuntimeError(
            "torchaudio or soundfile+librosa is required to load audio paths inside WhisperDPOCollator."
        )

    waveform, sr = sf.read(path, dtype="float32")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sr != sampling_rate:
        if librosa is None:
            raise RuntimeError(
                "Sample rate mismatch and librosa is unavailable for resampling. "
                "Install torchaudio or librosa to handle resampling."
            )
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sampling_rate)
    return torch.from_numpy(np.asarray(waveform, dtype=np.float32))


@dataclass
class WhisperDPOCollator:
    """
    Collate DPO triplets into Whisper-ready tensors.

    Input samples must contain the following keys:
        - "audio" or "audio_path": raw waveform information.
        - "chosen": preferred transcription string.
        - "rejected": dispreferred transcription string.

    Examples
    --------
    >>> collator = WhisperDPOCollator(processor)
    >>> batch = collator([sample0, sample1])
    >>> batch.keys()
    dict_keys(['input_features', 'attention_mask', 'labels_pos', 'labels_neg'])
    """

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

    def __call__(self, features: Iterable[Sample]) -> Batch:
        feature_list = list(features)
        if not feature_list:
            raise ValueError("WhisperDPOCollator received an empty batch.")

        # 1) Resolve waveforms for each sample.
        waveforms: List[torch.Tensor] = []
        for sample in feature_list:
            waveform = self._resolve_waveform(sample)
            waveforms.append(waveform)

        # 2) Convert to log-mel input features (with padding).
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
        batch: Batch = {
            "input_features": feat_inputs.input_features,  # [B, feat, frames]
        }
        if hasattr(feat_inputs, "attention_mask"):
            batch["attention_mask"] = feat_inputs.attention_mask

        # 3) Tokenize positive (chosen) and negative (rejected) transcripts.
        chosen_seqs = [sample["chosen"] for sample in feature_list]
        rejected_seqs = [sample["rejected"] for sample in feature_list]

        batch["labels_pos"] = self._tokenize_targets(chosen_seqs)
        batch["labels_neg"] = self._tokenize_targets(rejected_seqs)
        return batch

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _resolve_waveform(self, sample: Sample) -> torch.Tensor:
        """Extract mono waveform from various supported data formats."""
        if "audio" in sample:
            audio = sample["audio"]
            if isinstance(audio, dict):
                array = np.asarray(audio["array"], dtype=np.float32)
                sr = audio.get("sampling_rate", self.sampling_rate)
                waveform = torch.from_numpy(array)
            else:
                waveform = torch.as_tensor(audio, dtype=torch.float32)
                sr = sample.get("sampling_rate", self.sampling_rate)
        elif "audio_path" in sample:
            waveform = self.load_audio(sample["audio_path"], self.sampling_rate)
            sr = self.sampling_rate
        else:
            raise KeyError("Sample must contain either 'audio' or 'audio_path'.")

        if waveform.ndim > 1:
            waveform = _to_mono(waveform)
        if sr != self.sampling_rate:
            if torchaudio is None:
                raise RuntimeError(
                    "Audio sampling rate mismatch but torchaudio (for resampling) is unavailable."
                )
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            waveform = resampler(waveform)

        return waveform.float()

    def _tokenize_targets(self, sentences: List[str]) -> torch.Tensor:
        """Tokenize target sequences and replace padding with ignore index."""
        outputs = self._tokenizer(
            sentences,
            return_tensors="pt",
            padding="max_length",
            truncation=self.max_label_length is not None,
            max_length=self.max_label_length,
            return_attention_mask=True,
        )

        input_ids = outputs.input_ids
        attn_mask = outputs.attention_mask
        # Preserve EOS (within attention mask==1) and mask out only padded positions.
        input_ids = input_ids.masked_fill(attn_mask.eq(0), self.label_pad_token_id)
        return input_ids


__all__ = ["WhisperDPOCollator", "default_audio_loader"]
