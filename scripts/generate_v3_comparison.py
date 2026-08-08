from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from omnivoice import OmniVoice


BASE_MODEL = ("k2-fsa/OmniVoice", "c5fdb5ccb189668d56333f77ba2629f4cd7535f4")
V3_MODEL = (
    "2broke2code/serendib-omnivoice-finetuned-v3",
    "5857eb287f856364ce8c2440c8043cc42b1de791",
)
REFERENCE_AUDIO = Path("app/voices/female-004.wav")
REFERENCE_TEXT = (
    "Good morning sir, සර්ගේ vehicle insurance policy එක ලබන සතියෙන් expire වෙනවා. "
    "සර් කැමති නම් අපි දැන්ම ඒක renew කරන්න process එක පටන් ගන්න පුළුවන්"
)
SCRIPTS = {
    312: (
        20260816,
        "ඔබතුමාගේ data center එකේ cooling system එකේ warning alert එකක් ඇවිත් "
        "තියෙනවා Sir. අපි technician කෙනෙක්ව දැන්ම එවන්නම්.",
    ),
    330: (
        20260834,
        "අපේ CRM system එකෙන් Sir ගේ sales team එකේ performance එක live monitor "
        "කරන්න පුළුවන් dashboard එකක් හම්බවෙනවා.",
    ),
    369: (
        20261096,
        "ඔබතුමාගේ corporate tax එක online ගෙවන්න අදාළ reference number එක system "
        "එකට ඇතුළත් කරන්න Sir.",
    ),
}
DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32


def normalize_waveform(audio) -> np.ndarray:
    waveform = audio[0] if isinstance(audio, (list, tuple)) else audio
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu().numpy()
    waveform = np.squeeze(np.asarray(waveform))
    if waveform.ndim > 1:
        waveform = waveform[0]
    return waveform.astype(np.float32, copy=False)


def file_metrics(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    with wave.open(str(path), "rb") as audio_file:
        frames = audio_file.getnframes()
        sample_rate = audio_file.getframerate()
        channels = audio_file.getnchannels()
        sample_width = audio_file.getsampwidth()
    samples, _ = sf.read(path, dtype="float32")
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "duration_seconds": round(frames / sample_rate, 3),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bits": sample_width * 8,
        "peak": round(float(np.max(np.abs(samples))), 6),
        "rms": round(float(np.sqrt(np.mean(np.square(samples)))), 6),
    }


def generate_variant(
    label: str,
    model_id: str,
    revision: str,
    output_dir: Path,
) -> list[dict[str, object]]:
    model = OmniVoice.from_pretrained(
        model_id,
        revision=revision,
        device_map=DEVICE,
        dtype=DTYPE,
        load_asr=False,
    )
    sample_rate = getattr(model, "sampling_rate", 24000)
    results = []
    for script_id, (seed, text) in SCRIPTS.items():
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            audio = model.generate(
                language="si",
                text=text,
                ref_audio=str(REFERENCE_AUDIO),
                ref_text=REFERENCE_TEXT,
                num_step=20,
                speed=1.0,
            )
        output_path = output_dir / f"script_{script_id}_{label}.wav"
        sf.write(output_path, normalize_waveform(audio), sample_rate, subtype="PCM_16")
        results.append(
            {
                "script_id": script_id,
                "seed": seed,
                "text": text,
                "variant": label,
                "model_id": model_id,
                "revision": revision,
                "file": output_path.name,
                **file_metrics(output_path),
            }
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return results


def main() -> None:
    load_dotenv()
    output_dir = Path("comparison_v3")
    output_dir.mkdir(exist_ok=True)
    results = []
    for label, (model_id, revision) in ("base", BASE_MODEL), ("v3", V3_MODEL):
        results.extend(generate_variant(label, model_id, revision, output_dir))
    manifest = {
        "runtime_device": DEVICE,
        "runtime_dtype": str(DTYPE),
        "reference_audio": str(REFERENCE_AUDIO),
        "reference_text": REFERENCE_TEXT,
        "unseen_training_range": "Scripts 312, 330, and 369 are outside V3 IDs 1-311.",
        "results": results,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
