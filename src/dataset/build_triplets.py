"""
Utility script to build DPO triplets from the Zeroth-Korean dataset.

The workflow:
    1. Load the `Bingsu/zeroth-korean` dataset (test split by default).
    2. Use Whisper to transcribe the audio (rejected candidate).
    3. Combine ground-truth text (chosen) and Whisper prediction (rejected) into
       JSONL triplets consumable by `WhisperDPOCollator`.

This script is intentionally lightweight so it can be invoked via
`scripts/build_data.sh` or directly with `python -m src.dataset.build_triplets`.
"""

from __future__ import annotations

import os
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO triplets from Zeroth-Korean.")
    parser.add_argument("--split", default="test", help="Dataset split to use.")
    parser.add_argument("--dataset_name", default="Bingsu/zeroth-korean", help="HF dataset id.")
    parser.add_argument("--model_name", default="openai/whisper-tiny", help="Whisper base model.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap.")
    parser.add_argument(
        "--output_path",
        default="data/dpo_triplets/zeroth_test.jsonl",
        help="Destination JSONL file.",
    )
    parser.add_argument(
        "--audio_dir",
        default="data/processed/zeroth_test",
        help="Directory to store processed wav files.",
    )
    parser.add_argument("--device", default=None, help="torch device override (auto if None).")
    return parser.parse_args()


def prepare_model(model_name: str, device: Optional[str]) -> tuple:
    processor = WhisperProcessor.from_pretrained(model_name, language="ko", task="transcribe")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    LOGGER.info("Loaded %s on %s", model_name, device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ko", task="transcribe")
    return processor, model, forced_decoder_ids, device


def transcribe_batch(
    processor: WhisperProcessor,
    model: AutoModelForSpeechSeq2Seq,
    forced_decoder_ids,
    device: str,
    batch_audio: List[np.ndarray],
) -> List[str]:
    inputs = processor.feature_extractor(
        batch_audio,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    ).input_features.to(device)
    with torch.no_grad():
        generated_tokens = model.generate(
            input_features=inputs,
            forced_decoder_ids=forced_decoder_ids,
        )
    texts = processor.batch_decode(generated_tokens, skip_special_tokens=True)
    return texts


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    output_path = Path(args.output_path)
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    LOGGER.info("Loaded dataset %s split=%s size=%d", args.dataset_name, args.split, len(dataset))

    processor, model, forced_decoder_ids, device = prepare_model(args.model_name, args.device)
    sampling_rate = processor.feature_extractor.sampling_rate

    batch_size = 8 if device == "cuda" else 2
    buffer_audio: List[np.ndarray] = []
    buffer_indices: List[int] = []
    predictions: List[str] = [""] * len(dataset)

    for idx, sample in enumerate(tqdm(dataset, desc="Transcribing")):
        audio_array = sample["audio"]["array"]
        if sample["audio"]["sampling_rate"] != sampling_rate:
            raise ValueError("Unexpected sampling rate encountered in dataset.")
        buffer_audio.append(audio_array)
        buffer_indices.append(idx)
        if len(buffer_audio) >= batch_size:
            texts = transcribe_batch(processor, model, forced_decoder_ids, device, buffer_audio)
            for local_idx, text in zip(buffer_indices, texts):
                predictions[local_idx] = text
            buffer_audio.clear()
            buffer_indices.clear()
    if buffer_audio:
        texts = transcribe_batch(processor, model, forced_decoder_ids, device, buffer_audio)
        for local_idx, text in zip(buffer_indices, texts):
            predictions[local_idx] = text

    LOGGER.info("Writing triplets to %s", output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(dataset):
            audio_array = sample["audio"]["array"]
            output_wav = audio_dir / f"zeroth_{idx:05d}.wav"
            sf.write(output_wav, audio_array, sampling_rate, subtype="PCM_16")

            record = {
                "audio_path": str(output_wav),
                "chosen": sample["text"],
                "rejected": predictions[idx],
                "meta": {
                    "dataset": args.dataset_name,
                    "split": args.split,
                    "index": idx,
                    "model_rejected": args.model_name,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOGGER.info("Completed writing %d triplets.", len(dataset))


if __name__ == "__main__":
    main()
