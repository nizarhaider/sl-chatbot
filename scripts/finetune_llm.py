# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets>=4.0,<5",
#   "trl>=0.22",
#   "unsloth>=2026.4.2",
# ]
# ///
"""Fine-tune Gemma 4 E4B from the versioned Hugging Face dataset."""

import argparse
import os
import random

import torch
from datasets import load_dataset

BASE_MODEL = "unsloth/gemma-4-E4B-it"
DATASET_REPO = "2broke2code/serendib-sinhala-callcenter-sft-v1"
ADAPTER_REPO = "2broke2code/serendib-gemma-4-e4b-sinhala-callcenter-lora-v1"
SEED = 3407


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_REPO)
    parser.add_argument("--output", default=ADAPTER_REPO)
    parser.add_argument("--epochs", type=float, default=3)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required")

    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only

    random.seed(SEED)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    dataset = load_dataset(args.dataset, token=token)
    model, tokenizer = FastModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=1024,
        load_in_4bit=True,
        full_finetuning=False,
    )
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
                    messages, tokenize=False, add_generation_prompt=False
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
            output_dir="/workspace/serendib-gemma-e4b-lora",
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
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
            report_to=[],
            push_to_hub=True,
            hub_model_id=args.output,
            hub_private_repo=True,
            save_total_limit=2,
        ),
    )
    trainer = train_on_responses_only(trainer)
    result = trainer.train()
    trainer.push_to_hub(
        commit_message=f"Complete training: loss={result.metrics.get('train_loss')}"
    )
    print({"training_wall_clock_seconds": result.metrics.get("train_runtime")})
    print(f"Adapter: https://huggingface.co/{args.output}")


if __name__ == "__main__":
    main()
