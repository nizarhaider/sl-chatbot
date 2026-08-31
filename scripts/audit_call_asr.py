#!/usr/bin/env python3
"""Transcribe archived stereo call recordings with the production ASR model.

The archive stores caller audio in the left channel.  This script recreates the
production 40 ms energy VAD boundaries, then emits one JSON record per caller
turn for human error review.  It deliberately never uploads recordings or
transcripts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
import torch
from av.audio.resampler import AudioResampler
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from app.voice.config import (
    TURN_END_SILENCE_CHUNKS,
    TURN_INPUT_CHUNK_SIZE,
    TURN_MIN_AUDIO_MS,
    TURN_SILENCE_THRESHOLD,
    TURN_SPEECH_START_CHUNKS,
    TURN_SPEECH_START_THRESHOLD,
    WHISPER_MAX_NEW_TOKENS,
    WHISPER_MODEL,
)
from app.voice.vad import VadState, pcm_rms


def caller_pcm(path: Path) -> bytes:
    """Decode only the archived caller (left) channel as 16 kHz PCM16."""
    resampler = AudioResampler(format="s16", layout="mono", rate=16_000)
    chunks: list[bytes] = []
    with av.open(path) as container:
        stream = next(stream for stream in container.streams if stream.type == "audio")
        for frame in container.decode(stream):
            data = frame.to_ndarray()
            if data.ndim == 2 and data.shape[0] > 1:
                frame = av.AudioFrame.from_ndarray(data[:1], format=frame.format.name, layout="mono")
                frame.sample_rate = stream.rate
            chunks.extend(item.to_ndarray().tobytes() for item in resampler.resample(frame))
        chunks.extend(item.to_ndarray().tobytes() for item in resampler.resample(None))
    return b"".join(chunks)


def caller_turns(pcm: bytes) -> list[tuple[float, bytes]]:
    """Apply the production VAD thresholds to one caller channel."""
    state = VadState()
    turns: list[tuple[float, bytes]] = []
    for offset in range(0, len(pcm) - TURN_INPUT_CHUNK_SIZE + 1, TURN_INPUT_CHUNK_SIZE):
        chunk = pcm[offset : offset + TURN_INPUT_CHUNK_SIZE]
        rms = pcm_rms(chunk)
        if not state.is_speaking:
            if rms > TURN_SPEECH_START_THRESHOLD:
                state.add_candidate(chunk)
                if state.candidate_count >= TURN_SPEECH_START_CHUNKS:
                    state.promote_candidate()
            else:
                state.clear_candidate()
            continue
        if rms > TURN_SILENCE_THRESHOLD:
            state.add_speech(chunk)
            continue
        state.add_silence(chunk)
        if state.silence_chunks >= TURN_END_SILENCE_CHUNKS:
            turn = state.finish()
            if len(turn.pcm) / 2 / 16_000 * 1000 >= TURN_MIN_AUDIO_MS:
                turns.append((offset / 2 / 16_000, turn.pcm))
    return turns


def load_model(device: torch.device):
    dtype = torch.float16 if device.type == "mps" else torch.float32
    processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    model.eval()
    return processor, model, dtype


def transcribe(processor, model, dtype, device: torch.device, pcm: bytes) -> str:
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    inputs = processor(audio, sampling_rate=16_000, return_attention_mask=True, return_tensors="pt")
    with torch.inference_mode():
        ids = model.generate(
            inputs.input_features.to(device, dtype=dtype),
            attention_mask=inputs.attention_mask.to(device),
            language="sinhala",
            task="transcribe",
            max_new_tokens=WHISPER_MAX_NEW_TOKENS,
        )
    return processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; rerun with --device cpu")
    processor, model, dtype = load_model(device)
    records = []
    for recording in sorted(args.input_dir.glob("*.mp3")):
        for index, (end_seconds, pcm) in enumerate(caller_turns(caller_pcm(recording)), start=1):
            started = time.perf_counter()
            records.append({
                "recording": recording.name,
                "turn": index,
                "end_seconds": round(end_seconds, 3),
                "duration_seconds": round(len(pcm) / 2 / 16_000, 3),
                "transcript": transcribe(processor, model, dtype, device, pcm),
                "asr_seconds": round(time.perf_counter() - started, 3),
            })
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"turns={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
