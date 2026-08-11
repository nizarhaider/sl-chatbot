#!/usr/bin/env bash
# Run the official two-stage OmniVoice fine-tuning pipeline on a CUDA host.

set -euo pipefail

DATASET_REPO="${DATASET_REPO:-2broke2code/serendib-omnivoice-dataset-v5}"
DATASET_REVISION="${DATASET_REVISION:-5d2f4cc973f2c923a84607e60d746df1be2eb0dd}"
RUN_NAME="${RUN_NAME:-serendib-v5-repro}"
STEPS="${STEPS:-1312}"
WORKSPACE="${WORKSPACE:-/workspace}"
OMNIVOICE_DIR="$WORKSPACE/OmniVoice"
DATA_DIR="$WORKSPACE/$RUN_NAME-data"
TOKEN_DIR="$OMNIVOICE_DIR/data/$RUN_NAME/tokens"
OUTPUT_DIR="$WORKSPACE/$RUN_NAME"
SOURCE_REVISION="468e927ba3716cd8dd86421148dfb3046e9f9d7b"

log() { printf '▶ %s\n' "$*"; }
test -n "${HF_TOKEN:-}" || { echo "HF_TOKEN is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
test ! -e "$DATA_DIR" && test ! -e "$OUTPUT_DIR" || {
  echo "Data or output for $RUN_NAME already exists; choose a new RUN_NAME" >&2
  exit 1
}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

if [ ! -d "$OMNIVOICE_DIR/.git" ]; then
  git clone https://github.com/k2-fsa/OmniVoice.git "$OMNIVOICE_DIR"
fi
git -C "$OMNIVOICE_DIR" checkout -q "$SOURCE_REVISION"

if [ -d /venv/main ]; then
  source /venv/main/bin/activate
else
  uv venv "$WORKSPACE/omnivoice-venv"
  source "$WORKSPACE/omnivoice-venv/bin/activate"
fi
uv pip install -e "$OMNIVOICE_DIR"

mkdir -p "$DATA_DIR" "$OMNIVOICE_DIR/examples/config"
hf download "$DATASET_REPO" train.jsonl --type dataset \
  --revision "$DATASET_REVISION" --local-dir "$DATA_DIR" --quiet

log "Downloading only the files listed in train.jsonl; holdouts stay absent"
DATA_DIR="$DATA_DIR" DATASET_REPO="$DATASET_REPO" DATASET_REVISION="$DATASET_REVISION" python - <<'PY'
import json, os, wave
from pathlib import Path
from huggingface_hub import hf_hub_download

root = Path(os.environ["DATA_DIR"])
rows = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
seconds = 0.0
for row in rows:
    path = Path(hf_hub_download(
        repo_id=os.environ["DATASET_REPO"],
        repo_type="dataset",
        revision=os.environ["DATASET_REVISION"],
        filename=row["audio_path"],
        local_dir=root,
    ))
    with wave.open(str(path), "rb") as audio:
        seconds += audio.getnframes() / audio.getframerate()
    row["audio_path"] = str(path.resolve())
(root / "train.training.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
)
print(f"training_recordings={len(rows)}")
print(f"training_audio_minutes={seconds / 60:.3f}")
PY

TRAIN_CONFIG="$OMNIVOICE_DIR/examples/config/train_config_${RUN_NAME}.json"
DATA_CONFIG="$OMNIVOICE_DIR/examples/config/data_config_${RUN_NAME}.json"
STEPS="$STEPS" TOKEN_DIR="$TOKEN_DIR" TRAIN_CONFIG="$TRAIN_CONFIG" DATA_CONFIG="$DATA_CONFIG" python - <<'PY'
import json, os
train = {
    "llm_name_or_path": "Qwen/Qwen3-0.6B", "audio_vocab_size": 1025,
    "audio_mask_id": 1024, "num_audio_codebook": 8,
    "audio_codebook_weights": [8,8,6,6,4,4,2,2], "drop_cond_ratio": 0.1,
    "prompt_ratio_range": [0.0,0.3], "mask_ratio_range": [0.0,1.0],
    "language_ratio": 0.8, "use_pinyin_ratio": 0.0, "instruct_ratio": 0.0,
    "only_instruct_ratio": 0.0, "resume_from_checkpoint": None,
    "init_from_checkpoint": "k2-fsa/OmniVoice", "learning_rate": 1e-5,
    "weight_decay": 0.01, "max_grad_norm": 1.0, "steps": int(os.environ["STEPS"]),
    "seed": 42, "lr_scheduler_type": "cosine", "warmup_type": "ratio",
    "warmup_ratio": 0.03, "warmup_steps": 0, "batch_tokens": 512,
    "gradient_accumulation_steps": 8, "num_workers": 4, "mixed_precision": "bf16",
    "allow_tf32": True, "use_deepspeed": False, "deepspeed_config": None,
    "attn_implementation": "sdpa", "max_sample_tokens": 2000,
    "min_sample_tokens": 50, "max_batch_size": 1, "logging_steps": 25,
    "eval_steps": 250, "save_steps": 250, "keep_last_n_checkpoints": 2,
}
open(os.environ["TRAIN_CONFIG"], "w").write(json.dumps(train, indent=2) + "\n")
data = {"train": [{"manifest_path": [os.environ["TOKEN_DIR"] + "/train/data.lst"]}]}
open(os.environ["DATA_CONFIG"], "w").write(json.dumps(data, indent=2) + "\n")
PY

log "Tokenizing training audio"
CUDA_VISIBLE_DEVICES=0 python -m omnivoice.scripts.extract_audio_tokens \
  --input_jsonl "$DATA_DIR/train.training.jsonl" \
  --tar_output_pattern "$TOKEN_DIR/train/audios/shard-%06d.tar" \
  --jsonl_output_pattern "$TOKEN_DIR/train/txts/shard-%06d.jsonl" \
  --tokenizer_path eustlb/higgs-audio-v2-tokenizer \
  --nj_per_gpu 3 \
  --shuffle True | tee "$OUTPUT_DIR-tokenization.log"

log "Fine-tuning for $STEPS steps"
started="$(date +%s)"
accelerate launch \
  --gpu_ids 0 \
  --num_processes 1 \
  -m omnivoice.cli.train \
  --train_config "$TRAIN_CONFIG" \
  --data_config "$DATA_CONFIG" \
  --output_dir "$OUTPUT_DIR" | tee "$OUTPUT_DIR-training.log"
log "Training wall-clock seconds: $(( $(date +%s) - started ))"
log "Final output: $OUTPUT_DIR"
