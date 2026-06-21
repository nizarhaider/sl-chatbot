# WhatsApp Voice Agent Runtime Report

Generated from the active Vast.ai deployment on 2026-06-21.

Remote instance inspected:

```text
Host: 203.121.204.241
SSH port: 41990
Remote repo path: /workspace/sl-chatbot
Runtime commit: 75792b1 Enhance conversation history management and update response generation to include context
Runtime timestamp checked: 2026-06-21T10:19:11+00:00
```

## Executive Summary

The current WhatsApp voice agent is running end-to-end on a Vast.ai GPU instance. The live call path completed successfully with no logged application errors, no empty Gemma responses, no empty Whisper transcripts, and no TTS failures.

The latest test call shows that the system can:

- Receive and answer WhatsApp Calling SDP offers.
- Establish an aiortc peer connection.
- Receive inbound WhatsApp audio.
- Segment speech using local RMS VAD.
- Transcribe Sinhala and Sinhala/English mixed speech with local Whisper.
- Maintain short per-call conversation history across turns.
- Generate local Gemma responses without hosted LLM APIs.
- Synthesize Sinhala speech through RealtimeTTS OmniVoice.
- Stream PCM audio back into the outbound aiortc track.

The strongest current bottleneck is ASR quality for noisy or code-mixed Sinhala speech. Gemma and TTS are responding reliably, but several transcripts are garbled enough that the agent asks the caller to repeat. When the transcript is clear enough, the agent answers property questions correctly and uses the available mock Homelands property data.

## Current Architecture

The current production voice path is:

```text
WhatsApp Cloud webhook
  -> FastAPI /webhook
  -> WhatsApp Calling SDP offer
  -> aiortc RTCPeerConnection
  -> inbound WhatsApp Opus audio track
  -> local audio resampling to 16 kHz mono PCM
  -> local RMS VAD turn detection
  -> local Whisper STT
  -> local Gemma 4 12B Q4 via llama.cpp / llama-cpp-python
  -> RealtimeTTS OmniVoice synthesis
  -> outbound 48 kHz stereo PCM buffering
  -> aiortc outbound audio track
  -> WhatsApp call audio
```

The assistant brain remains local Gemma. No hosted LLM API is in the voice path.

## Runtime Hardware

Live hardware reported by the active Vast instance:

| Component | Current Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080 |
| GPU VRAM total | 16,303 MiB |
| GPU VRAM used at inspection | 13,054 MiB |
| NVIDIA driver | 580.159.03 |
| GPU temperature | 45 C |
| GPU power draw at inspection | 17.61 W |
| GPU utilization at inspection | 0% |
| CPU | AMD Ryzen 5 7500F 6-Core Processor |
| CPU cores / threads | 6 cores / 12 threads |
| CPU max frequency | 5074 MHz |
| System memory | 31 GiB total |
| System memory used | 5.7 GiB |
| System memory available | 25 GiB |
| Swap | 8.0 GiB |
| Root/workspace disk | 32 GiB total, 19 GiB used, 14 GiB available |

The GPU memory footprint is high but expected for the current stack. At inspection, about 13.1 GiB of 16.3 GiB VRAM was in use after loading Whisper, Gemma Q4, and OmniVoice. This leaves limited headroom on 16 GB GPUs, especially if model context, TTS settings, or concurrent calls increase.

## Runtime Software

Live runtime versions:

| Package / Runtime | Version |
| --- | --- |
| Python | 3.12.3 |
| torch | 2.7.1+cu128 |
| transformers | 5.8.1 |
| llama-cpp-python | 0.3.31 |
| RealtimeTTS | 0.7.1 |
| OmniVoice | 0.1.5 |
| aiortc | 1.14.0 |
| FastAPI | 0.129.0 |

Active process shape:

```text
tmux session: sl-ngrok
  ngrok http 8081 --log=stdout

tmux session: sl-webhook
  .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8081 --env-file .env
```

The webhook process is started with these runtime exports:

```text
WHISPER_DEVICE=cuda
GEMMA_MODEL_REPO=google/gemma-4-12B-it-qat-q4_0-gguf
GEMMA_N_GPU_LAYERS=-1
GEMMA_CONTEXT_TOKENS=4096
GEMMA_PREWARM=true
```

## Model Configuration

Current model path and settings observed in logs:

| Layer | Current Configuration |
| --- | --- |
| ASR | SPEAK-ASR/whisper-medium-si-merged |
| ASR device | cuda |
| ASR dtype | torch.float16 |
| LLM | google/gemma-4-12B-it-qat-q4_0-gguf |
| LLM file | gemma-4-12b-it-qat-q4_0.gguf |
| LLM backend | llama-cpp-python / llama.cpp |
| LLM GPU layers | -1, all supported layers in VRAM |
| LLM context | 4096 tokens |
| LLM batch | 512 tokens |
| TTS | RealtimeTTS OmniVoice |
| TTS device | cuda:0 |
| TTS dtype | float16 |
| TTS step schedule | observed setup uses low-latency OmniVoice settings |

Conversation memory is now enabled in the deployed commit. The agent keeps the last 6 user/assistant turns per active call and passes them into Gemma as chat history before the latest transcript.

## Startup And Prewarm

Relevant startup events:

```text
2026-06-21 10:10:11.982 Loading Hugging Face Whisper ASR model: SPEAK-ASR/whisper-medium-si-merged (device: cuda dtype: torch.float16)
2026-06-21 10:11:50.666 Loading local Gemma model: ... gemma-4-12b-it-qat-q4_0.gguf n_gpu_layers=-1 n_ctx=4096 n_batch=512
2026-06-21 10:11:51.894 Local Gemma model loaded in 1227 ms
2026-06-21 10:12:24.610 WEBHOOK_VERIFIED
2026-06-21 10:12:29.160 WEBHOOK_VERIFIED
```

Gemma itself loaded quickly once the model was already available on disk. The longer startup phase is dominated by Whisper and OmniVoice initialization and dependency/model availability.

## Latest Call Summary

Latest call window:

```text
Connect event:   2026-06-21 10:13:04.581 UTC
Connected state: 2026-06-21 10:13:05.812 UTC
Terminate event: 2026-06-21 10:15:56.931 UTC
Approx duration: 2 minutes 52 seconds
Completed turns: 11
```

Connection setup was healthy:

```text
Received call event: connect
Received audio track from WhatsApp
Connection state: connecting
Connection state: connected
...
Received call event: terminate
Connection state: closed
```

No errors, tracebacks, empty Gemma responses, empty transcripts, or TTS failures were found in the inspected logs.

## Latency Stats

Stats from 11 completed caller turns in the latest call:

| Metric | Average | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| Whisper STT | 321 ms | 135 ms | 549 ms |
| Gemma LLM | 592 ms | 277 ms | 1,071 ms |
| TTS wall time | 946 ms | 381 ms | 1,844 ms |
| Synthesized audio duration | 9,389 ms | 2,720 ms | 18,400 ms |
| Post-VAD response time | 1,860 ms | 817 ms | 3,190 ms |
| Total turn time | 4,822 ms | 2,714 ms | 8,311 ms |

Interpretation:

- STT latency is strong. The average transcription time was about 0.32 seconds.
- Gemma latency is strong for a local 12B Q4 model. The average response generation time was about 0.59 seconds.
- TTS generation wall time is also strong. It averaged under 1 second.
- The largest user-perceived delay is dominated by how much audio the TTS response produces. Some responses generated 10-18 seconds of audio, so the system can feel slower even when compute latency is low.
- The current system is viable for single-call testing on a 16 GB RTX 5080, but VRAM headroom is limited.

## Turn-Level Latency

| Turn | Transcript Summary | STT | LLM | TTS Wall | TTS Audio | Total |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `sinhala` | 184 ms | 447 ms | 381 ms | 3,560 ms | 3,125 ms |
| 2 | Garbled Sinhala question about what the agent does | 519 ms | 368 ms | 742 ms | 5,000 ms | 7,715 ms |
| 3 | Clearer repeat: what do you do? | 427 ms | 640 ms | 1,162 ms | 12,760 ms | 5,409 ms |
| 4 | Garbled property/memory phrase | 240 ms | 372 ms | 748 ms | 5,000 ms | 3,902 ms |
| 5 | Partially garbled properties request | 373 ms | 494 ms | 1,140 ms | 10,200 ms | 4,830 ms |
| 6 | Where are properties located? | 302 ms | 1,044 ms | 1,844 ms | 18,400 ms | 5,725 ms |
| 7 | Horizon Residencies price | 549 ms | 1,071 ms | 1,229 ms | 17,280 ms | 8,311 ms |
| 8 | `හරි` | 135 ms | 277 ms | 404 ms | 2,720 ms | 2,714 ms |
| 9 | What is your name? | 244 ms | 561 ms | 1,147 ms | 9,720 ms | 4,495 ms |
| 10 | Garbled phrase | 171 ms | 442 ms | 744 ms | 5,000 ms | 3,038 ms |
| 11 | What else can you do? | 387 ms | 801 ms | 867 ms | 13,640 ms | 3,777 ms |

## Conversation Quality

The agent handled the initial language selection correctly:

```text
Caller: sinhala
Agent: ආයුබෝවන්! ඔබට කොහොමද මම උදව් කරන්න පුළුවන්?
```

The agent asked for repetition when the transcript was unclear:

```text
Transcript: අ කාලමාට පොඩී ප්‍රශ්නය. ඔය මොකද කරන්නේ?
Agent: සමාවන්න, මට ඔබ පවසන දේ පැහැදිලි නැත. කරුණාකර නැවත පවසන්න.
```

When the caller repeated the question more clearly, the agent recovered:

```text
Transcript: මං ඇහුවේ ඔය මොකක්ද කරන්නේ කියලා
Agent: මම Homelands Properties ආයතනයේ සහායකයෙක්...
```

The agent eventually handled a broad property availability question:

```text
Transcript: properties කොහේත් තියෙන්නේ මේ
Agent: අප සතුව ප්‍රධාන වශයෙන් පහත ප්‍රදේශවල දේපළ පවතී:
1. මාලබේ (Horizon Residencies)
2. පීලියන්දල (Lakeview Villas)
3. කුරුනෑගල (Green Acres)
4. දෙහිවල (Ocean Breeze Apartments)
```

The agent answered the Horizon Residencies price question correctly:

```text
Transcript: හෝමේ අඩ horizon residence සීස් එකට කීයත වගේ වෙන්නේ price එක
Agent: Horizon Residencies හි ... මිල රුපියල් මිලියන 28 සිට ආරම්භ වේ...
```

The new conversation history appears to be working. The later property answer benefited from prior turns and the agent no longer behaved as if every turn was a completely new call.

## Current Strengths

- End-to-end call pipeline is operational.
- Local Gemma response latency is very good.
- Local Whisper transcription latency is very good.
- TTS generation wall time is fast enough for interactive use.
- History-enabled turn handling improves continuity across caller turns.
- The system stays within the local-only assistant requirement.
- The agent can recover after unclear ASR when the caller repeats.
- The deployed RTX 5080 16 GB instance can run the current single-call stack.

## Current Weaknesses

- Sinhala ASR quality is the main functional bottleneck. Some transcripts are distorted, especially with code-mixed Sinhala/English and informal speech.
- VRAM headroom is tight on 16 GB. The inspected runtime used about 13.1 GiB of 16.3 GiB.
- Some responses are too long for a phone call. Even though TTS compute is fast, the synthesized audio can be 10-18 seconds long.
- Broad or garbled property questions still sometimes trigger repetition or generic clarification.
- Current stats are based on one recent call. More calls are needed before treating the averages as stable production benchmarks.

## Operational Notes

Current healthy call path checklist:

```text
1. WEBHOOK_VERIFIED appears.
2. Received call event: connect appears.
3. Received audio track from WhatsApp appears.
4. Connection state becomes connected.
5. Greeting TTS completes.
6. Turn VAD detects speech start and speech end.
7. Turn transcript appears.
8. Turn response appears.
9. RealtimeTTS complete appears.
10. Call terminate closes the peer connection.
```

Useful logs:

```bash
tail -f /workspace/sl-chatbot/run_logs/important.log
tail -f /workspace/sl-chatbot/run_logs/webhook.log
```

Useful runtime checks:

```bash
nvidia-smi
tmux ls
ps -eo pid,ppid,cmd | grep -E 'uvicorn|ngrok' | grep -v grep
curl -sS http://127.0.0.1:8081/
```

## Recommendations

Near-term:

- Continue collecting call logs from realistic Sinhala callers.
- Track ASR failure examples separately and compare alternate Whisper settings or models.
- Keep the 6-turn history limit for now. It is working and has low complexity.
- Consider shortening responses at the generation layer later if call pacing feels slow.

Infrastructure:

- 16 GB VRAM is workable for one active call, but leaves limited headroom.
- For more stable experimentation, 24 GB GPUs such as RTX 3090/4090 are safer.
- For repeated fresh deployments, build a prebuilt image or wheelhouse to avoid rebuilding CUDA dependencies and `llama-cpp-python` every time.

