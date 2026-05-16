import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_MODELS = [
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "microsoft/Phi-4-mini-instruct",
]

DEFAULT_PROMPT = (
    "You are an SLT Mobitel phone support agent. "
    "Reply in one short sentence to this customer question: "
    "'My fiber internet is down since this morning. What should I do first?'"
)


@dataclass
class BenchmarkResult:
    model: str
    device: str
    dtype: str
    load_seconds: float
    llm_seconds: float
    tts_seconds: float
    total_seconds: float
    output_chars: int
    output_preview: str
    audio_seconds: float


def choose_device() -> tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value)


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a concise SLT Mobitel support agent. "
                "Keep replies under 18 words and avoid bullet points."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def load_model(model_id: str, device: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()
    return tokenizer, model


def generate_text(tokenizer, model, prompt: str, device: str) -> str:
    messages = build_messages(prompt)
    chat_template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **chat_template_kwargs,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
    inputs = tokenizer(rendered, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


async def benchmark_model(model_id: str, prompt: str) -> BenchmarkResult:
    device, dtype = choose_device()

    load_started = time.perf_counter()
    tokenizer, model = await asyncio.to_thread(load_model, model_id, device, dtype)
    load_seconds = time.perf_counter() - load_started

    llm_started = time.perf_counter()
    text = await asyncio.to_thread(generate_text, tokenizer, model, prompt, device)
    llm_seconds = time.perf_counter() - llm_started

    from app.services.tts import get_tts_service

    tts_service = get_tts_service()
    tts_started = time.perf_counter()
    synthesized = await tts_service.synthesize(text)
    tts_seconds = time.perf_counter() - tts_started

    audio_seconds = len(synthesized.pcm) / 2 / synthesized.sample_rate

    del model
    if device in {"cuda", "mps"}:
        if device == "cuda":
            torch.cuda.empty_cache()
        else:
            torch.mps.empty_cache()

    return BenchmarkResult(
        model=model_id,
        device=device,
        dtype=str(dtype).replace("torch.", ""),
        load_seconds=round(load_seconds, 3),
        llm_seconds=round(llm_seconds, 3),
        tts_seconds=round(tts_seconds, 3),
        total_seconds=round(llm_seconds + tts_seconds, 3),
        output_chars=len(text),
        output_preview=text[:160],
        audio_seconds=round(audio_seconds, 3),
    )


async def main():
    load_env_file()
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    results = []

    for model_id in models:
        print(f"Benchmarking {model_id}...")
        result = await benchmark_model(model_id, args.prompt)
        results.append(result)

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False))
        return

    for result in results:
        print()
        print(result.model)
        print(
            f"  load={result.load_seconds}s "
            f"llm={result.llm_seconds}s "
            f"tts={result.tts_seconds}s "
            f"total={result.total_seconds}s "
            f"audio={result.audio_seconds}s"
        )
        print(f"  reply={result.output_preview}")


if __name__ == "__main__":
    asyncio.run(main())
