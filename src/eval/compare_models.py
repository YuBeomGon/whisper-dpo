"""
Utility script to compare baseline vs preference-optimized Whisper models.

Given a JSONL dataset containing {audio_path, chosen} fields, this module
decodes the audio with two checkpoints and reports WER, along with sample
transcriptions for manual inspection.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from jiwer import wer
from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor

LOGGER = logging.getLogger(__name__)
DEFAULT_SR = 16000


def read_triplets(json_path: Path, sample_limit: int | None = None) -> Tuple[List[str], List[np.ndarray]]:
    """
    Load references and audio waveforms from a JSONL triplet file.

    Only the `chosen` transcript and `audio_path` fields are required.
    The audio is converted to mono float32 and resampled to 16 kHz if needed.
    """
    refs: List[str] = []
    wavs: List[np.ndarray] = []
    with json_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if sample_limit is not None and idx >= sample_limit:
                break
            record = json.loads(line)
            refs.append(record["chosen"])

            wav, sr = sf.read(record["audio_path"], dtype="float32")
            if wav.ndim == 2:
                wav = wav.mean(axis=1)
            if sr != DEFAULT_SR:
                raise ValueError(f"Expected sampling rate {DEFAULT_SR}, but got {sr}. Preprocess ahead of time.")
            wavs.append(wav.astype(np.float32))
    LOGGER.info("Loaded %d examples from %s", len(refs), json_path)
    return refs, wavs


def decode_batch(
    model_id: str | Path,
    refs: Sequence[str],
    wavs: Sequence[np.ndarray],
    device: str,
    language: str,
    task: str,
) -> Tuple[float, List[str]]:
    """
    Decode the given audio list with Whisper and return WER + transcripts.

    Parameters
    ----------
    model_id:
        Hugging Face model id or local path.
    refs:
        Reference transcripts aligned with `wavs`.
    wavs:
        List of wavform arrays (mono float32 at DEFAULT_SR).
    device:
        torch device string ("cuda" or "cpu").
    language:
        Whisper language code (e.g. "ko").
    task:
        Whisper task ("transcribe" or "translate").
    """
    processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    model.to(device)
    model.eval()

    preds: List[str] = []
    for wav in wavs:
        inputs = processor(wav, sampling_rate=DEFAULT_SR, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(**inputs)
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        preds.append(text)
    score = wer(refs, preds)
    return score, preds


def format_samples(
    refs: Sequence[str],
    baseline_preds: Sequence[str],
    dpo_preds: Sequence[str],
    sample_indices: Iterable[int],
) -> str:
    lines = []
    for idx in sample_indices:
        if idx >= len(refs):
            break
        lines.append(f"[#{idx}] ref: {refs[idx]}")
        lines.append(f"[#{idx}] base: {baseline_preds[idx]}")
        lines.append(f"[#{idx}] dpo : {dpo_preds[idx]}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs DPO Whisper checkpoints on WER.")
    parser.add_argument("--data", required=True, help="JSONL with {audio_path, chosen}.")
    parser.add_argument("--baseline", required=True, help="Baseline model id or local path.")
    parser.add_argument("--model", required=True, help="DPO fine-tuned model path.")
    parser.add_argument("--language", default="ko", help="Whisper language code.")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--device", default=None, help="torch device override (auto if omitted).")
    parser.add_argument("--limit", type=int, help="Optional cap on number of samples.")
    parser.add_argument("--samples", type=int, nargs="*", default=[0, 1, 2], help="Sample indices to print.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Using device: %s", device)

    refs, wavs = read_triplets(Path(args.data), sample_limit=args.limit)

    base_wer, base_preds = decode_batch(
        args.baseline, refs, wavs, device=device, language=args.language, task=args.task
    )
    dpo_wer, dpo_preds = decode_batch(
        args.model, refs, wavs, device=device, language=args.language, task=args.task
    )

    print(f"Baseline WER: {base_wer:.4f}")
    print(f"DPO      WER: {dpo_wer:.4f}")

    if args.samples:
        print("\n--- Sample Transcriptions ---")
        print(format_samples(refs, base_preds, dpo_preds, args.samples))


if __name__ == "__main__":
    main()

