"""
Supervised fine-tuning (SFT) launcher for Whisper on JSONL datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from datasets import Dataset
from transformers import (
    AutoModelForSpeechSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperProcessor,
    set_seed,
)

from ..dataset import PreferenceTripletDataset, WhisperSFTCollator


def _configure_environment(args: argparse.Namespace) -> None:
    if getattr(args, "hf_home", None):
        Path(args.hf_home).mkdir(parents=True, exist_ok=True)
        import os

        os.environ.setdefault("HF_HOME", args.hf_home)


LOGGER = logging.getLogger(__name__)


def _load_yaml(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with supervised transcripts.")
    parser.add_argument("--config", type=str, help="Optional YAML config file.")
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--eval_dataset", type=str)
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny")
    parser.add_argument("--output_dir", type=str, default="outputs/sft")
    parser.add_argument("--language", type=str, default="ko")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--evaluation_strategy", choices=["no", "steps", "epoch"], default="steps")
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--hf_home", type=str)
    parser.add_argument("--report_to", nargs="*", default=["tensorboard"])
    known_args, _ = parser.parse_known_args()
    defaults = _load_yaml(known_args.config)
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args()
    if args.train_dataset is None:
        raise ValueError("--train_dataset must be provided (via CLI or config).")
    return args


def load_sft_dataset(path: str) -> Dataset:
    triplets = PreferenceTripletDataset(path=path)
    entries = []
    for sample in triplets.to_list():
        entries.append({"audio_path": sample["audio_path"], "text": sample["chosen"]})
    return Dataset.from_list(entries)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("SFT args: %s", json.dumps(vars(args), indent=2))

    _configure_environment(args)
    set_seed(args.seed)

    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language=args.language,
        task=args.task,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_name)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    if forced_decoder_ids is not None:
        model.config.forced_decoder_ids = forced_decoder_ids
        if hasattr(model, "generation_config"):
            model.generation_config.forced_decoder_ids = forced_decoder_ids

    train_dataset = load_sft_dataset(args.train_dataset)
    eval_dataset = load_sft_dataset(args.eval_dataset) if args.eval_dataset else None

    data_collator = WhisperSFTCollator(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy=args.evaluation_strategy if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16 and not args.fp16,
        predict_with_generate=True,
        generation_max_length=448,
        remove_unused_columns=False,
        report_to=args.report_to,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    trainer.save_model()
    processor.save_pretrained(Path(args.output_dir) / "processor")


if __name__ == "__main__":
    main()
