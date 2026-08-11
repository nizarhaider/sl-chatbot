#!/usr/bin/env python3
"""Small local Gradio UI for the pinned V5 voice model."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import torch
from dotenv import load_dotenv

from app.models import OmniVoiceTTS


def best_device() -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda:0", "float16"
    if torch.backends.mps.is_available():
        return "mps", "float32"
    return "cpu", "float32"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local SerendibAI V5 voice inference"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    load_dotenv()
    device, dtype = best_device()
    tts = OmniVoiceTTS(device=device, dtype=dtype)

    def synthesize(text: str, seed: int):
        return tts.sample_rate, tts.synthesize(text, int(seed))

    demo = gr.Interface(
        fn=synthesize,
        inputs=[
            gr.Textbox(label="Sinhala / Singlish text", lines=4),
            gr.Number(label="Seed", value=42, precision=0),
        ],
        outputs=gr.Audio(label="Generated speech", type="numpy"),
        title="SerendibAI OmniVoice V5",
        description=f"Runs locally on {device}. The model and reference audio are pinned and cached from Hugging Face.",
        flagging_mode="never",
    )
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port
    )


if __name__ == "__main__":
    main()
