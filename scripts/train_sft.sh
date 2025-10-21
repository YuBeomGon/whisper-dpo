#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HF_HOME=${HF_HOME:-"$ROOT_DIR/.cache/huggingface"}

ACCELERATE_BIN=${ACCELERATE_BIN:-accelerate}

"$ACCELERATE_BIN" launch -m src.modeling.train_sft "$@"
