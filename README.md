# Whisper DPO Hallucination Suppression

Preference-aligned fine-tuning pipeline for OpenAI Whisper focused on reducing hallucinations that appear during long non-speech segments. The project combines supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) on Zeroth-Korean audio, and provides tooling to benchmark baseline vs. tuned checkpoints.

## 1. Project Overview

- **Problem context**: Whisper sometimes produces hallucinated text (loops, outro phrases, false statements) when fed silence/noise. The accompanying `poc.txt` summarises literature indicating the social risk and key mitigation strategies.
- **Approach**:
  1. Collect preference pairs: `{audio_path, chosen (reference transcript), rejected (hallucinated output)}`.
  2. Run supervised fine-tuning to anchor the model to the target domain.
  3. Apply DPO on the preference pairs to penalise hallucinated completions.
  4. Evaluate all three checkpoints (baseline, SFT, DPO) on the same dataset with WER and sample transcriptions.

## 2. Repository Structure

```
src/
  dataset/
    collator_dpo.py      # Audio + chosen/rejected batching for DPO
    collator_sft.py      # Audio + transcript batching for SFT
    triplet_dataset.py   # JSONL loader for preference pairs
  modeling/
    train_sft.py         # Supervised fine-tuning entry point
    train_dpo.py         # DPO training entry point
    dpo_trainer_whisper.py
    losses.py
  eval/
    compare_models.py    # Baseline vs SFT/DPO WER comparison
configs/
  train_sft.yaml         # Default SFT hyperparameters
  train_dpo.yaml         # Default DPO hyperparameters
scripts/
  build_data.sh          # Generate preference JSONL from Zeroth dataset
  train_sft.sh
  train_dpo.sh
  eval_all.sh
data/
  dpo_triplets/          # Preference JSONL (generated)
  processed/             # Augmented audio (generated)
outputs/
  sft/                   # SFT checkpoints (generated)
  dpo/                   # DPO checkpoints (generated)
GEMINI.md                # Implementation design notes
poc.txt                  # Literature-backed motivation
README.md                # This file
```

## 3. Environment Setup

```bash
conda create -n whisper-dpo python=3.10
conda activate whisper-dpo

# Install PyTorch (choose the index that matches your CUDA)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Useful environment variables:

- `HF_HOME=./.cache/huggingface` keeps Hugging Face downloads inside the repo.
- `TOKENIZERS_PARALLELISM=false` avoids tokenizer warning spam.

## 4. Data Pipeline

1. **Zeroth-Korean preference generation** (baseline hallucinations used as rejected samples):

   ```bash
   ./scripts/build_data.sh \
     --split train \
     --model_name openai/whisper-tiny \
     --output_path data/dpo_triplets/zeroth_train.jsonl \
     --audio_dir data/processed/zeroth_train

   ./scripts/build_data.sh \
     --split test \
     --model_name openai/whisper-tiny \
     --output_path data/dpo_triplets/zeroth_test.jsonl \
     --audio_dir data/processed/zeroth_test
   ```

   Each JSONL line contains `audio_path`, `chosen`, `rejected`, and metadata needed by the collators.

2. **Dataset format**: the loaders expect JSONL fields `audio_path`, `chosen`, `rejected`. For SFT we reuse the same file but only consume the `chosen` transcripts.

## 5. Training Workflows

### 5.1 Supervised Fine-Tuning (SFT)

```bash
./scripts/train_sft.sh --config configs/train_sft.yaml
```

Key settings (`configs/train_sft.yaml`):

- `per_device_train_batch_size: 2`
- `gradient_accumulation_steps: 8`
- `learning_rate: 5e-6`
- `num_train_epochs: 1`
- `fp16: true`
- `gradient_checkpointing: false`
- `hf_home: .cache/huggingface`
- `report_to: [tensorboard]`

The script writes the checkpoint to `outputs/sft/...` and saves the matching `WhisperProcessor` for later evaluation.

### 5.2 Direct Preference Optimization (DPO)

```bash
./scripts/train_dpo.sh --config configs/train_dpo.yaml
```

Defaults (`configs/train_dpo.yaml`):

- `model_name: openai/whisper-tiny` (replace with SFT checkpoint by editing the config if you want SFT→DPO)
- `beta: 0.1`
- `calibration_weight: 0.05`
- `per_device_train_batch_size: 2`
- `gradient_accumulation_steps: 8`
- `learning_rate: 5e-6`
- `num_train_epochs: 1`
- `fp16: true`
- `gradient_checkpointing: false`
- `hf_home: .cache/huggingface`

Evaluation dataset can be specified via `eval_dataset` (defaults to `zeroth_test.jsonl`). The trainer copies the reference model internally and computes policy/reference log-prob differences.

## 6. Evaluation & Ablation

### 6.1 Quick Baseline vs. Tuned Comparison

```bash
# Baseline OpenAI checkpoint
./scripts/eval_all.sh \
  --data data/dpo_triplets/zeroth_test.jsonl \
  --baseline openai/whisper-tiny \
  --model openai/whisper-tiny \
  --samples 0 12

# SFT checkpoint
./scripts/eval_all.sh \
  --data data/dpo_triplets/zeroth_test.jsonl \
  --baseline openai/whisper-tiny \
  --model outputs/sft/whisper_tiny_zeroth \
  --samples 0 12

# DPO checkpoint (with SFT as baseline)
./scripts/eval_all.sh \
  --data data/dpo_triplets/zeroth_test.jsonl \
  --baseline outputs/sft/whisper_tiny_zeroth \
  --model outputs/dpo/whisper_tiny_zeroth \
  --samples 0 12 42
```

`eval_all.sh` invokes `python -m src.eval.compare_models`, which computes WER for both models, prints sample transcriptions (reference, baseline, tuned), and supports additional flags:

- `--limit 100` to restrict evaluation set size.
- `--device cuda:0` or `--device cpu`.
- `--language` and `--task` to override processor defaults if needed.

### 6.2 Metrics Beyond WER

Add insertion rate (IR), deletion rate (DR), and hallucination-specific metrics by extending `src/eval/metrics.py`. The scaffolding exists (see `GEMINI.md` section 5 for recommended formulas).

## 7. Common Issues & Mitigations

- **Model generates endless repeated tokens (`emá…`)**: Ensure collators preserve EOS tokens. `collator_dpo.py` and `collator_sft.py` pad only the true pad positions (`attention_mask == 0`) with `-100`.
- **Whisper requires 3000 mel frames**: Both collators pad waveforms to Whisper’s nominal sample length (`n_samples`) via `padding="max_length"` and `max_length`.
- **`torchaudio` missing**: The collators fall back to `soundfile` + `librosa` but resampling needs at least one of these libraries. Install `torchaudio` to avoid slow path.
- **HF Hub access blocked**: Set `HF_HOME` and cache models in advance (`huggingface-cli download ...`).

## 8. Next Steps

1. **Mask-DPO / Token-level DPO**: Narrow the penalty to hallucinated spans (`Mask-DPO`) to avoid punishing correct tokens inside rejected transcripts.
2. **Hyperparameter sweeps**: Lower β (e.g., 0.02‒0.05) and calibrate length penalty to check for WER regression vs. hallucination suppression gains.
3. **Evaluation expansions**: Implement IR, DR, FPM-NS, and Bag-of-Hallucinations hit rate to monitor hallucination trade-offs separately from deletion/omission errors.
4. **Dataset improvements**: Improve chosen/rejected quality with human review or automated filters, and keep SFT checkpoints as the warm start for DPO.

## 9. References

- `poc.txt`: Problem definition, related literature, and risk assessment.
- GEMINI execution notes (`GEMINI.md`) document the implementation plan, hyperparameter recommendations, and pipeline warnings.

---

This README summarises the current training and evaluation tooling. Contributions are welcome—please open an issue or PR with new datasets, metric scripts, or ablation results.

