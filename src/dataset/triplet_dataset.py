"""
Dataset utilities for Whisper preference optimization.

This module exposes a lightweight JSONL dataset loader that returns the fields
expected by `WhisperDPOCollator`: audio (or audio_path), chosen, rejected, and
optional metadata.  It keeps dependencies minimal so it can be reused inside
scripts or data builders without pulling in `datasets`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset


@dataclass
class TripletSample:
    audio_path: Optional[str]
    audio: Optional[torch.Tensor]
    chosen: str
    rejected: str
    meta: Dict

    def as_dict(self) -> Dict:
        data: Dict = {"chosen": self.chosen, "rejected": self.rejected}
        if self.audio is not None:
            data["audio"] = self.audio.cpu().numpy().tolist()
        if self.audio_path is not None:
            data["audio_path"] = self.audio_path
        if self.meta:
            data["meta"] = self.meta
        return data


class PreferenceTripletDataset(Dataset):
    """
    Load DPO triplets stored as JSON Lines.

    Parameters
    ----------
    path:
        JSONL file with keys `audio_path` or `audio`, `chosen`, `rejected`, `meta`.
    preload_audio:
        If True, audio waveforms will be loaded eagerly using `audio_loader`.
    audio_loader:
        Callable that accepts `(audio_path, sampling_rate)` and returns a tensor.
    sampling_rate:
        Sampling rate forwarded to `audio_loader`.
    """

    def __init__(
        self,
        path: str,
        preload_audio: bool = False,
        audio_loader: Optional[Callable[[str, int], torch.Tensor]] = None,
        sampling_rate: int = 16000,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Triplet dataset not found: {self.path}")
        self.preload_audio = preload_audio
        self.audio_loader = audio_loader
        self.sampling_rate = sampling_rate
        self._samples: List[TripletSample] = []
        self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                audio_tensor = None
                audio_path = record.get("audio_path")
                if self.preload_audio:
                    if audio_path is None:
                        raise ValueError(
                            "preload_audio=True requires each sample to include 'audio_path'."
                        )
                    if self.audio_loader is None:
                        raise ValueError(
                            "An audio_loader must be provided when preload_audio=True."
                        )
                    audio_tensor = self.audio_loader(audio_path, self.sampling_rate)
                sample = TripletSample(
                    audio_path=audio_path,
                    audio=audio_tensor,
                    chosen=record["chosen"],
                    rejected=record["rejected"],
                    meta=record.get("meta", {}),
                )
                self._samples.append(sample)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict:
        return self._samples[idx].as_dict()

    def __iter__(self) -> Iterator[Dict]:
        for sample in self._samples:
            yield sample.as_dict()

    def to_list(self) -> List[Dict]:
        return [sample.as_dict() for sample in self._samples]


__all__ = ["PreferenceTripletDataset"]
