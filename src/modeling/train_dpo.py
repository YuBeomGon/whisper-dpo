"""
Command-line entry point for Whisper DPO fine-tuning.

This script wires together the custom `WhisperDPOTrainer`, dataset collator,
and preference triplet loader so that training can be launched via
`accelerate` or plain Python.
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
from transformers import AutoModelForSpeechSeq2Seq, WhisperProcessor, set_seed
from trl import DPOConfig

from ..dataset import PreferenceTripletDataset, WhisperDPOCollator
from . import WhisperDPOTrainer, clone_reference_model, maybe_unfreeze_lora_peft


LOGGER = logging.getLogger(__name__)


def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Whisper with DPO on preference triplets.")
    parser.add_argument("--config", type=str, help="Optional YAML config file with defaults.")

    # Dataset / data loader
    parser.add_argument("--train_dataset", type=str, help="Path to train JSONL triplets.")
    parser.add_argument("--eval_dataset", type=str, help="Optional eval JSONL triplets.")
    parser.add_argument("--dataset_preload_audio", action="store_true", help="Preload audio tensors into memory.")

    # Model / processor
    parser.add_argument("--model_name", type=str, default="openai/whisper-small")
    parser.add_argument("--language", type=str, default="ko", help="Language code for decoder prompt ids.")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to resume checkpoint.")

    # Optimization hyperparameters
    parser.add_argument("--output_dir", type=str, default="outputs/dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--calibration_weight", type=float, default=0.0)
    parser.add_argument("--length_normalization", choices=["average", "sum"], default="average")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Logging / saving
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--evaluation_strategy", choices=["no", "steps", "epoch"], default="steps")
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--report_to", nargs="*", default=["tensorboard"], help="Reporting targets (e.g., tensorboard, wandb).")

    # Adapter / PEFT toggles
    parser.add_argument("--lora_r", type=int, default=0, help="Enable LoRA if r > 0. (adapter setup handled externally)")

    # Misc
    parser.add_argument("--hf_home", type=str, help="Override HF cache dir e.g. ./.cache/huggingface")
    parser.add_argument("--debug_print_config", action="store_true", help="Print resolved config and exit.")

    # First pass to detect config
    known_args, _ = parser.parse_known_args()
    defaults = _load_yaml(known_args.config)
    if defaults:
        parser.set_defaults(**defaults)
    return parser.parse_args()


def _resolve_dataset(path: str, preload_audio: bool) -> Dataset:
    triplets = PreferenceTripletDataset(path=path, preload_audio=preload_audio)
    LOGGER.info("Loaded dataset %s with %d samples", path, len(triplets))
    hf_dataset = Dataset.from_list(triplets.to_list())
    return hf_dataset


def _configure_environment(args: argparse.Namespace) -> None:
    if args.hf_home:
        Path(args.hf_home).mkdir(parents=True, exist_ok=True)
        torch_set = torch.__dict__.get("set_float32_matmul_precision")
        if torch_set:
            torch_set("high")
        import os

        os.environ.setdefault("HF_HOME", args.hf_home)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Parsed args: %s", json.dumps(vars(args), indent=2))

    if args.debug_print_config:
        return

    _configure_environment(args)
    set_seed(args.seed)

    if args.train_dataset is None:
        raise ValueError("--train_dataset must be provided (path to JSONL triplets).")

    processor = WhisperProcessor.from_pretrained(
        args.model_name,
        language=args.language,
        task=args.task,
    )

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language,
        task=args.task,
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_name)
    if forced_decoder_ids is not None:
        model.config.forced_decoder_ids = forced_decoder_ids
        if hasattr(model, "generation_config"):
            model.generation_config.forced_decoder_ids = forced_decoder_ids

    if args.lora_r > 0:
        LOGGER.warning("LoRA setup is not implemented in this CLI. Configure adapters externally.")
        maybe_unfreeze_lora_peft(model)

    reference_model = clone_reference_model(model)

    train_dataset = _resolve_dataset(args.train_dataset, args.dataset_preload_audio)
    eval_dataset = (
        _resolve_dataset(args.eval_dataset, args.dataset_preload_audio)
        if args.eval_dataset
        else None
    )

    data_collator = WhisperDPOCollator(processor=processor)

    config_kwargs = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
        seed=args.seed,
        report_to=args.report_to,
        beta=args.beta,
        do_eval=eval_dataset is not None,
        eval_strategy=args.evaluation_strategy if eval_dataset is not None else "no",
        disable_dropout=False,
    )
    if not torch.cuda.is_available():
        config_kwargs["use_cpu"] = True
        config_kwargs["fp16"] = False
        config_kwargs["bf16"] = False

    dpo_config = DPOConfig(**config_kwargs)

    trainer_kwargs = dict(
        model=model,
        args=dpo_config,
        length_normalization=args.length_normalization,
        calibration_loss_weight=args.calibration_weight,
        ref_model=reference_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )
    trainer = WhisperDPOTrainer(**trainer_kwargs)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()
    processor.save_pretrained(Path(args.output_dir) / "processor")


if __name__ == "__main__":
    main()
