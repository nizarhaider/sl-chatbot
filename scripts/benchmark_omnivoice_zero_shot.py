import argparse
import json
import statistics
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


DEFAULT_REF_AUDIO = (
    "training_custom_tts/datasets/openslr_sinhala/extracted/sin_2241_9304867830.wav"
)
DEFAULT_REF_TEXT = (
    "සෙලාන් සමග ඊට විශාල වෙනසක් නොතිබුණත් කමර්ෂල් බැංකුවට එතරම් "
    "මිළක් ඇත්තේ මන්දැයි හිතා ගන්නවත් බැහැ"
)
DEFAULT_TEXTS = [
    "හායි, හෝම්ලෑන්ඩ්ස් ආයතනය හා සම්බන්ධ වූවාට ස්තූතියි.",
    "මම තරුශි. මට අද ඔබට කොහොමද සහය වෙන්න පුලුවන්.",
    "ඔබගේ ඉල්ලීම පරීක්ෂා කරලා ඉක්මනින්ම උදව් කරන්නම්.",
]


@dataclass
class RunResult:
    label: str
    text_chars: int
    wall_seconds: float
    audio_seconds: float
    rtf: float


def choose_device() -> tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    return "cpu", torch.float32


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def audio_duration(audio, sample_rate: int) -> float:
    return len(audio[0]) / sample_rate


def sync_device(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--num-step", type=int, default=8)
    parser.add_argument("--speed", type=float, default=1.18)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    ref_audio = Path(args.ref_audio)
    if not ref_audio.exists():
        raise FileNotFoundError(ref_audio)

    from omnivoice import OmniVoice

    device, dtype = choose_device()
    load_started = time.perf_counter()
    model = OmniVoice.from_pretrained(args.model, device_map=device, dtype=dtype)
    sync_device(device)
    load_seconds = time.perf_counter() - load_started

    if args.warmup:
        _ = model.generate(
            text=DEFAULT_TEXTS[0],
            ref_audio=str(ref_audio),
            ref_text=args.ref_text,
            num_step=args.num_step,
            speed=args.speed,
        )
        sync_device(device)

    results: list[RunResult] = []
    for index, text in enumerate(DEFAULT_TEXTS, start=1):
        started = time.perf_counter()
        audio = model.generate(
            text=text,
            ref_audio=str(ref_audio),
            ref_text=args.ref_text,
            num_step=args.num_step,
            speed=args.speed,
        )
        sync_device(device)
        wall_seconds = time.perf_counter() - started
        seconds = audio_duration(audio, 24000)
        results.append(
            RunResult(
                label=f"run_{index}",
                text_chars=len(text),
                wall_seconds=round(wall_seconds, 3),
                audio_seconds=round(seconds, 3),
                rtf=round(wall_seconds / seconds, 3),
            )
        )

    payload = {
        "model": args.model,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "torch": torch.__version__,
        "num_step": args.num_step,
        "speed": args.speed,
        "ref_audio": str(ref_audio),
        "ref_audio_seconds": round(wav_duration(ref_audio), 3),
        "load_seconds": round(load_seconds, 3),
        "runs": [asdict(result) for result in results],
        "mean_wall_seconds": round(statistics.mean(r.wall_seconds for r in results), 3),
        "mean_audio_seconds": round(statistics.mean(r.audio_seconds for r in results), 3),
        "mean_rtf": round(statistics.mean(r.rtf for r in results), 3),
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(
        f"Loaded {payload['model']} on {device}/{payload['dtype']} "
        f"in {payload['load_seconds']}s"
    )
    print(
        f"ref_audio={payload['ref_audio']} "
        f"({payload['ref_audio_seconds']}s) num_step={args.num_step} speed={args.speed}"
    )
    for result in results:
        print(
            f"{result.label}: wall={result.wall_seconds}s "
            f"audio={result.audio_seconds}s rtf={result.rtf}"
        )
    print(
        f"mean: wall={payload['mean_wall_seconds']}s "
        f"audio={payload['mean_audio_seconds']}s rtf={payload['mean_rtf']}"
    )


if __name__ == "__main__":
    main()
