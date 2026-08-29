# Optimal Voice Runtime Settings

This is the production baseline for the SerendibAI WhatsApp voice agent. Keep
the canonical values in `app/voice/config.py`; this document is the human
readable architecture and parameter record.

## Architecture

```text
WhatsApp Cloud webhook
  -> aiortc peer connection
  -> 16 kHz mono PCM
  -> local RMS VAD
  -> SPEAK-ASR Sinhala Whisper
  -> local llama.cpp / Gemma tool loop
  -> RealtimeTTS OmniVoice
  -> 48 kHz stereo aiortc output track
  -> WhatsApp call
```

All inference stays local to the GPU runtime. Pinecone backs property retrieval;
Neon persists appointments and call data. WhatsApp and WebRTC are transport only.

## Target runtime

| Component | Setting |
| --- | --- |
| GPU baseline | NVIDIA RTX 3090, 24 GB VRAM |
| Python runtime | Locked `uv` environment |
| LLM server | llama.cpp OpenAI-compatible endpoint at `http://127.0.0.1:8000/v1` |
| Application | FastAPI / aiortc, supervised as `sl-webhook` |

## ASR

| Parameter | Value |
| --- | --- |
| Model | `SPEAK-ASR/whisper-medium-si-merged` |
| Device | `cuda:0` |
| Precision | FP16 on CUDA |
| Forced language | Sinhala |
| Task | Transcribe |
| Maximum new tokens | 64 |
| Input | 16 kHz mono PCM |

## Gemma

| Parameter | Value |
| --- | --- |
| Model | `unsloth/gemma-4-E4B-it-GGUF` |
| Quantization/runtime | Unsloth UD-Q4_K_XL through local llama.cpp |
| Temperature | 1.0 |
| Maximum output tokens | 128 |
| Tool choice | Auto; real `search_properties`, `book_appointment`, and `send_whatsapp_message` tools |
| Prewarm | Enabled |

The model can trigger a property search through its OpenAI-style tool call. If it
explicitly says it will search but omits that call, the pipeline sends the real
`search_properties` request using the accumulated caller history.

## Turn control and barge-in

| Parameter | Value |
| --- | --- |
| Input chunk | 40 ms / 1,280 bytes of PCM16 |
| Speech-start RMS | 900 |
| Speech-start duration | 5 chunks / 200 ms |
| Speech continuation RMS | 1,000 |
| End silence | 18 chunks / 720 ms |
| Minimum turn audio | 320 ms |
| Barge-in RMS | 900 |
| Barge-in duration | 5 chunks / 200 ms |
| Greeting delay | 0.5 s |
| Playback echo tail | 1.5 s |

On a confirmed interruption, the output buffer is cleared and queued TTS chunks
are checked again against the playback generation before being sent. Isolated
language-menu echoes after language selection are rejected rather than added to
caller context.

## OmniVoice

| Parameter | Value |
| --- | --- |
| Model | `2broke2code/serendib-omnivoice-finetuned-v2` |
| Device / precision | `cuda:0` / FP16 |
| Reference audio | `app/voices/female-004.wav` |
| Reference language | Sinhala |
| Steps schedule | `20,20` |
| Speed | 1.0 |
| Output synthesis rate | 24 kHz mono PCM |
| WebRTC output | 48 kHz stereo PCM |
| Prewarm | Enabled |

## Regression expectations

`scripts/voice_regression.py` replays the supplied recordings through the real
VAD, ASR, Gemma, tools, and OmniVoice components. It asserts language selection,
property-tool behavior when the model declares search intent, caller-context
retention, echo rejection, barge-in, and a maximum 5-second caller-end to first
audible-audio latency. Artifacts remain outside Git under
`run_logs/voice_regressions/`.
