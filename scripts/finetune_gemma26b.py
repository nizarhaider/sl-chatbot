# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "datasets>=4.0,<5",
#   "huggingface-hub>=1.5,<2",
#   "trackio",
#   "transformers==5.5.0",
#   "tokenizers>=0.22,<=0.23",
#   "trl",
#   "unsloth",
# ]
# ///
"""Fine-tune Gemma 4 26B-A4B for the SerendibAI phone agent."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from trl import SFTConfig, SFTTrainer

BASE_MODEL = "unsloth/gemma-4-26B-A4B-it"
DATASET_REPO = "2broke2code/serendib-sinhala-callcenter-sft-v1"
ADAPTER_REPO = "2broke2code/serendib-gemma-4-26b-a4b-sinhala-callcenter-lora-v1"
GGUF_REPO = "2broke2code/serendib-gemma-4-26b-a4b-sinhala-callcenter-gguf-v1"
OUTPUT_DIR = "/workspace/serendib-gemma-4-26b-a4b-lora"
SEED = 3407


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Publish the completed local adapter and export its merged Q4 GGUF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    api = HfApi(token=token)
    for repo_id in (ADAPTER_REPO, GGUF_REPO):
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True)

    if args.postprocess_only:
        publish_and_export(api, token)
        return

    gpu = torch.cuda.get_device_properties(0)
    print(
        {
            "gpu": gpu.name,
            "gpu_vram_gib": round(gpu.total_memory / 1024**3, 2),
            "base_model": BASE_MODEL,
            "dataset": DATASET_REPO,
        }
    )

    dataset = load_dataset(DATASET_REPO, token=token)
    validate_dataset(dataset)
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        dtype=None,
        max_seq_length=1024,
        load_in_4bit=True,
        full_finetuning=False,
        token=token,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4-thinking")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=SEED,
    )

    def format_rows(batch: dict) -> dict:
        return {
            "text": [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                ).removeprefix("<bos>")
                for messages in batch["messages"]
            ]
        }

    formatted = dataset.map(format_rows, batched=True)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted["train"],
        eval_dataset=formatted["validation"],
        args=SFTConfig(
            output_dir=OUTPUT_DIR,
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=3.0,
            learning_rate=5e-5,
            warmup_steps=20,
            logging_steps=5,
            eval_strategy="steps",
            eval_steps=25,
            save_strategy="epoch",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            max_length=1024,
            seed=SEED,
            report_to="trackio",
            project="serendibai-llm",
            run_name="gemma-4-26b-a4b-sinhala-callcenter-v1",
            push_to_hub=False,
            save_total_limit=2,
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    started = time.perf_counter()
    result = trainer.train()
    wall_seconds = time.perf_counter() - started
    evaluation = next(
        (
            metrics
            for metrics in reversed(trainer.state.log_history)
            if "eval_loss" in metrics
        ),
        {},
    )
    trainer.save_model(OUTPUT_DIR)
    print(
        {
            "training_wall_seconds": round(wall_seconds, 1),
            "train_metrics": result.metrics,
            "eval_metrics": evaluation,
            "peak_reserved_vram_gib": round(
                torch.cuda.max_memory_reserved() / 1024**3, 2
            ),
        }
    )

    print("Restarting in a clean process for Hub publication and GGUF export")
    os.execv(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), "--postprocess-only"],
    )


def publish_and_export(api: HfApi, token: str) -> None:
    output = Path(OUTPUT_DIR)
    if not (output / "adapter_model.safetensors").is_file():
        raise RuntimeError(f"Completed adapter not found in {OUTPUT_DIR}")

    print("Uploading completed adapter from a clean process")
    api.upload_large_folder(
        repo_id=ADAPTER_REPO,
        repo_type="model",
        folder_path=output,
        private=True,
        ignore_patterns=["checkpoint-*", ".git*"],
        print_report_every=60,
    )

    print("Loading completed adapter for merged Q4_K_M GGUF export")
    model, tokenizer = FastModel.from_pretrained(
        model_name=OUTPUT_DIR,
        dtype=None,
        max_seq_length=1024,
        load_in_4bit=True,
        full_finetuning=False,
        token=token,
    )
    print("Exporting and uploading merged Q4_K_M GGUF")
    model.push_to_hub_gguf(
        GGUF_REPO,
        tokenizer,
        quantization_method="q4_k_m",
        token=token,
    )
    print({"adapter": ADAPTER_REPO, "gguf": GGUF_REPO})


def validate_dataset(dataset) -> None:
    if set(dataset) != {"train", "validation"}:
        raise ValueError(f"Unexpected dataset splits: {set(dataset)}")
    expected_rows = {"train": 935, "validation": 106}
    for split, expected in expected_rows.items():
        if len(dataset[split]) != expected:
            raise ValueError(
                f"Expected {expected} {split} rows, found {len(dataset[split])}"
            )
        for row in dataset[split]:
            messages = row["messages"]
            if not isinstance(messages, list) or len(messages) < 3:
                raise ValueError(f"Invalid messages in {split}")
            if [item["role"] for item in messages[:2]] != ["system", "user"]:
                raise ValueError(f"Invalid role order in {split}")
            if messages[-1]["role"] != "assistant":
                raise ValueError(f"Example does not end with assistant in {split}")


if __name__ == "__main__":
    main()
