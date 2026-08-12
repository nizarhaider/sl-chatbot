# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets>=4.0,<5",
#   "huggingface-hub>=0.36,<2",
#   "trl>=0.22",
#   "unsloth>=2026.4.2",
# ]
# ///
"""Prepare and fine-tune the Sinhala call-center LLM on a CUDA host.

The script builds grounded caller/agent pairs from the human-written voice
scripts, mixes in a small MIT-licensed Sinhala language-retention sample, and
trains a Gemma 4 E4B LoRA adapter. Outputs are pushed to private Hub repos.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
import random
import re
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi

ACCOUNT = "2broke2code"
BASE_MODEL = "unsloth/gemma-4-E4B-it"
TEACHER_MODEL = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
DATASET_REPO = f"{ACCOUNT}/serendib-sinhala-callcenter-sft-v1"
ADAPTER_REPO = f"{ACCOUNT}/serendib-gemma-4-e4b-sinhala-callcenter-lora-v1"
LANGUAGE_SOURCE = "ihalage/sinhala-instruction-finetune-large"
SEED = 3407

SYSTEM_PROMPT = (
    "You are a concise Sri Lankan phone agent. Identify the language of the caller's latest "
    "message and reply in that exact language; never default to Sinhala for Tamil or English. "
    "Keep replies short enough to speak aloud. Use only supplied facts, act immediately when "
    "enough information is present, and ask one brief question only when information is missing."
)

TOOL_SYSTEM_PROMPT = SYSTEM_PROMPT + " " + (
    "Property facts and viewing appointments are available only through tools. Never invent "
    "inventory, prices, availability, or confirmations. Emit exactly one <tool_call> JSON block "
    "when a property search or booking is required, without introductory or spoken text."
)


def text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text.strip()}]


def message(role: str, text: str) -> dict:
    return {"role": role, "content": text_content(text)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripts-csv", type=Path)
    parser.add_argument("--dataset-repo", default=DATASET_REPO)
    parser.add_argument("--adapter-repo", default=ADAPTER_REPO)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--max-scripts", type=int)
    parser.add_argument("--language-samples", type=int, default=200)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not args.train_only and not args.scripts_csv:
        parser.error("--scripts-csv is required unless --train-only is used")
    if args.prepare_only and args.train_only:
        parser.error("choose only one of --prepare-only or --train-only")
    return args


def require_environment() -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return token


def load_voice_scripts(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"ID", "Category", "Script"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain {sorted(required)}")
    rows = [row for row in rows if row["Script"].strip()]
    return rows[:limit] if limit else rows


def extract_json(text: str) -> dict | None:
    for candidate in re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, flags=re.DOTALL):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if all(isinstance(payload.get(key), str) and payload[key].strip() for key in ("caller_1", "caller_2", "known_facts")):
            return payload
    return None


def teacher_prompt(row: dict[str, str]) -> str:
    source = json.dumps(
        {"category": row["Category"], "agent_response": row["Script"]},
        ensure_ascii=False,
    )
    return f"""Create grounded supervised data for a Sri Lankan call-center model.

Source: {source}

Return one JSON object with exactly these string fields:
- caller_1: a natural short caller utterance that makes the agent response appropriate
- caller_2: a distinct paraphrase of caller_1
- known_facts: a concise English statement containing every fact the agent is allowed to state

Caller utterances should use the same Sinhala/English code-switching style as the response.
Do not copy the agent response into either caller field. Do not add markdown."""


@torch.inference_mode()
def generate_teacher_pairs(rows: list[dict[str, str]], seed: int) -> list[dict]:
    from unsloth import FastModel

    print(f"Loading teacher: {TEACHER_MODEL}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=TEACHER_MODEL,
        max_seq_length=1024,
        load_in_4bit=True,
        full_finetuning=False,
    )
    tokenizer.padding_side = "left"
    model.eval()
    prepared: list[dict] = []
    failures = 0
    batch_size = 4
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        conversations = [
            [
                message("system", "Return valid JSON only."),
                message("user", teacher_prompt(row)),
            ]
            for row in batch
        ]
        inputs = tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=240,
            do_sample=False,
            use_cache=True,
        )
        generated = tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        for row, raw in zip(batch, generated, strict=True):
            payload = extract_json(raw)
            if payload is None:
                failures += 1
                continue
            for variant in ("caller_1", "caller_2"):
                prepared.append(
                    {
                        "source": "serendib_voice_scripts",
                        "source_id": f"voice-{row['ID']}",
                        "messages": [
                            message("system", f"{SYSTEM_PROMPT}\n\nKnown facts: {payload['known_facts']}"),
                            message("user", payload[variant]),
                            message("assistant", row["Script"]),
                        ],
                    }
                )
        print(f"Prepared {min(offset + batch_size, len(rows))}/{len(rows)} scripts", flush=True)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Teacher generation complete: {len(prepared)} examples, {failures} skipped scripts")
    if len(prepared) < len(rows):
        raise RuntimeError("Too many teacher rows failed JSON validation")
    random.Random(seed).shuffle(prepared)
    return prepared


def language_examples(count: int, seed: int) -> list[dict]:
    stream = load_dataset(LANGUAGE_SOURCE, split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=5_000)
    examples = []
    for index, row in enumerate(stream.take(count)):
        question = str(row.get("question_prompt", "")).strip()
        answer = str(row.get("response_prompt", "")).strip()
        if not question or not answer:
            continue
        examples.append(
            {
                "source": LANGUAGE_SOURCE,
                "source_id": f"language-{index}",
                "messages": [
                    message("system", "Reply clearly and concisely in Sinhala."),
                    message("user", question),
                    message("assistant", answer),
                ],
            }
        )
    return examples


def tool_examples() -> list[dict]:
    locations = ["Malabe", "Piliyandala", "Kurunegala", "Dehiwala"]
    property_types = ["apartment", "villa", "land"]
    caller_templates = [
        "{location} වල {property_type} එකක් හොයනවා.",
        "මට {location} area එකෙන් {property_type} options බලන්න ඕනේ.",
        "Do you have a {property_type} in {location}?",
    ]
    rows: list[dict] = []
    index = 0
    for location in locations:
        for property_type in property_types:
            for template in caller_templates:
                arguments = {"location": location, "property_type": property_type}
                tool_call = f'<tool_call>{json.dumps({"name": "search_properties", "arguments": arguments}, separators=(",", ":"))}</tool_call>'
                rows.append(
                    {
                        "source": "serendib_tool_examples",
                        "source_id": f"tool-search-{index}",
                        "messages": [
                            message("system", TOOL_SYSTEM_PROMPT),
                            message("user", template.format(location=location, property_type=property_type)),
                            message("assistant", tool_call),
                        ],
                    }
                )
                index += 1

    booking_names = ["Nimal", "Amaya", "Kavindu", "Fathima"]
    for index, name in enumerate(booking_names):
        appointment = f"2026-08-{20 + index}T{10 + index}:00:00+05:30"
        arguments = {
            "property_id": "horizon-residencies-malabe",
            "customer_name": name,
            "appointment_at": appointment,
        }
        tool_call = f'<tool_call>{json.dumps({"name": "book_appointment", "arguments": arguments}, separators=(",", ":"))}</tool_call>'
        rows.append(
            {
                "source": "serendib_tool_examples",
                "source_id": f"tool-book-{index}",
                "messages": [
                    message("system", TOOL_SYSTEM_PROMPT),
                    message("user", f"මගේ නම {name}. Horizon Residencies බලන්න {20 + index} වෙනිදා appointment එකක් දාන්න."),
                    message("assistant", tool_call),
                ],
            }
        )

    behavior_rows = [
        ("සිංහල", "මම SerendibAI කෙනෙක්, Homelands Properties වෙනුවෙන් කතා කරනවා. ඔබට කොහොමද උදව් කරන්න ඕනේ?"),
        ("English", "I'm calling from SerendibAI on behalf of Homelands Properties. How can I help?"),
        ("ඔකේ", "හරි, ඔබට කොහොමද උදව් කරන්න ඕනේ?"),
        ("හරි", "හරි, ඔබට කොහොමද උදව් කරන්න ඕනේ?"),
        ("Okay", "Sure. How can I help?"),
        ("Yes", "Sure. How can I help?"),
    ]
    for index, (caller, agent) in enumerate(behavior_rows):
        rows.append(
            {
                "source": "serendib_behavior_examples",
                "source_id": f"behavior-{index}",
                "messages": [
                    message("system", TOOL_SYSTEM_PROMPT),
                    message("user", caller),
                    message("assistant", agent),
                ],
            }
        )

    named_properties = [
        ("Ocean Breeze", "Ocean Breeze Apartments"),
        ("Horizon Residencies", "Horizon Residencies"),
        ("Green Acres", "Green Acres"),
        ("Lakeview Villas", "Lakeview Villas"),
    ]
    for index, (caller_name, query) in enumerate(named_properties):
        arguments = {"query": query}
        tool_call = f'<tool_call>{json.dumps({"name": "search_properties", "arguments": arguments}, separators=(",", ":"))}</tool_call>'
        rows.append(
            {
                "source": "serendib_tool_examples",
                "source_id": f"tool-named-{index}",
                "messages": [
                    message("system", TOOL_SYSTEM_PROMPT),
                    message("user", f"{caller_name} ගැන විස්තර ඕනේ."),
                    message("assistant", tool_call),
                ],
            }
        )
    return rows


def validate_and_split(rows: list[dict], seed: int) -> DatasetDict:
    unique: dict[str, dict] = {}
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError(f"Invalid messages in {row.get('source_id')}")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"Example must end with assistant: {row.get('source_id')}")
        roles = [item.get("role") for item in messages]
        if roles[:2] != ["system", "user"]:
            raise ValueError(f"Invalid role order: {row.get('source_id')}")
        key = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        unique.setdefault(key, row)

    grouped: dict[str, list[dict]] = {}
    for row in unique.values():
        grouped.setdefault(row["source_id"], []).append(row)
    group_ids = sorted(grouped)
    random.Random(seed).shuffle(group_ids)
    validation_ids = set(group_ids[: max(1, round(len(group_ids) * 0.1))])
    train = [row for group_id in group_ids if group_id not in validation_ids for row in grouped[group_id]]
    validation = [row for group_id in group_ids if group_id in validation_ids for row in grouped[group_id]]
    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(validation)
    print({"train": len(train), "validation": len(validation), "unique_groups": len(group_ids)})
    return DatasetDict({"train": Dataset.from_list(train), "validation": Dataset.from_list(validation)})


def push_dataset(dataset: DatasetDict, repo_id: str, token: str) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    dataset.push_to_hub(repo_id, private=True, token=token)
    card = f"""---
license: other
language:
- si
- en
task_categories:
- text-generation
---

# SerendibAI Sinhala Call-Center SFT V1

Private supervised fine-tuning data for a concise Sinhala-English phone agent.

Sources:
- SerendibAI human-written call-center voice scripts (owned by SerendibAI).
- A small language-retention sample from `{LANGUAGE_SOURCE}` (MIT).
- Synthetic property-tool routing examples generated from the runtime schema.

Real production call transcripts are excluded from training and reserved for evaluation.
"""
    api.upload_file(
        path_or_fileobj=io.BytesIO(card.encode()),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Document training dataset",
    )
    print(f"Dataset: https://huggingface.co/datasets/{repo_id}")


def prepare_dataset(args: argparse.Namespace, token: str) -> None:
    rows = load_voice_scripts(args.scripts_csv, args.max_scripts)
    generated = generate_teacher_pairs(rows, args.seed)
    mixed = generated + language_examples(args.language_samples, args.seed) + tool_examples()
    dataset = validate_and_split(mixed, args.seed)
    push_dataset(dataset, args.dataset_repo, token)


def train(args: argparse.Namespace, token: str) -> None:
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset(args.dataset_repo, token=token)
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
        random_state=args.seed,
    )

    def format_rows(batch: dict) -> dict:
        texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False).removeprefix("<bos>")
            for messages in batch["messages"]
        ]
        return {"text": texts}

    formatted = dataset.map(format_rows, batched=True)
    config = SFTConfig(
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
        seed=args.seed,
        report_to=[],
        run_name="gemma-4-e4b-sinhala-callcenter-v1",
        project="serendibai-llm",
        push_to_hub=True,
        hub_model_id=args.adapter_repo,
        hub_private_repo=True,
        hub_strategy="every_save",
        save_total_limit=2,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=formatted["train"],
        eval_dataset=formatted["validation"],
        args=config,
    )
    trainer = train_on_responses_only(trainer)
    result = trainer.train()
    evaluation = next(
        (metrics for metrics in reversed(trainer.state.log_history) if "eval_loss" in metrics),
        {},
    )
    trainer.push_to_hub(
        commit_message=(
            f"Complete training: loss={result.metrics.get('train_loss')} "
            f"eval_loss={evaluation.get('eval_loss')}"
        )
    )
    print({"train_metrics": result.metrics, "eval_metrics": evaluation})
    print(f"Adapter: https://huggingface.co/{args.adapter_repo}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    token = require_environment()
    if not args.train_only:
        prepare_dataset(args, token)
    if not args.prepare_only:
        train(args, token)


if __name__ == "__main__":
    main()
