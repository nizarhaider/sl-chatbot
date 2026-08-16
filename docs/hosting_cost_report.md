# Voice Runtime Hosting Cost Report

Snapshot date: 2026-08-16

## Recommendation

Use two distinct sizing rules for the Gemma 4 26B-A4B voice agent:

- **32 GB VRAM is the hard runtime floor.** Select the cheapest compatible,
  verified listing at or above this capacity. It should hold the Q4 LLM,
  Whisper, OmniVoice, KV cache, and useful operating headroom on one GPU. The
  final requirement must still be confirmed with a full-stack measurement.
- **48 GB VRAM is the training and validation tier.** Use it for LoRA training,
  adapter merging, GGUF conversion, and the first end-to-end deployment. It is
  not automatically required for steady-state hosting.

The previous 4B voice runtime used about 13.1 GiB of VRAM with a 4.8 GiB GGUF.
Replacing that file with the 16.8 GB 26B-A4B Q4 model puts the
combined workload near or above a 24 GB card once CUDA overhead and KV cache
are included. The deployer therefore rejects configurations below 32 GB.

## Cheapest Live Vast.ai Listings

The deployer's compatibility and reliability query was run against live
on-demand listings with an 80 GB disk on 2026-08-16:

| Tier | Cheapest observed GPU | Hourly total | 730-hour month |
| --- | --- | ---: | ---: |
| 32 GB+ | Quadro RTX 8000 48 GB | $0.25111 | $183.31 |
| 48 GB+ | Quadro RTX 8000 48 GB | $0.25111 | $183.31 |

There was no cheaper compatible 32 GB listing at the snapshot, so the 32 GB
query selected a 48 GB Quadro RTX 8000. This older Turing card is inexpensive
and has ample memory, but a newer A6000, A40, L40, or RTX 6000 Ada may deliver
better latency. The A6000 48 GB contract used for the current training run costs
about $0.50074/hour with a 200 GB disk, or $365.54 for 730 continuous hours.

Vast.ai prices and availability change continuously. Totals include the disk
selected by the query but exclude usage-based network transfer. See Vast.ai's
[pricing documentation](https://docs.vast.ai/guides/instances/pricing).

## Training Result and Validation Status

The Gemma 4 26B-A4B response-only LoRA completed 351 optimizer steps over three
epochs on an RTX A6000. Training took 9,121 seconds (2 hours 32 minutes), with a
final training loss of 1.113 and validation loss of 1.246. The private adapter,
16.8 GB `Q4_K_M` GGUF, and `training_results.json` are stored on Hugging Face.

The training host was destroyed after the artifacts were verified. The full
Whisper + LLM + OmniVoice stack has not yet been measured with this model, so
record the following before selecting the long-running tier:

| Measurement | Required result |
| --- | --- |
| Idle full-stack VRAM | Fits with several GiB of headroom |
| Peak turn VRAM | No CUDA OOM during ASR, generation, or TTS |
| LLM first-token and total latency | Suitable for a phone conversation |
| End-to-end turn latency | Acceptable during an actual WhatsApp call |
| Disk usage | Leaves room for model cache and logs |

If peak usage is at most roughly 27-28 GiB, a 32 GB production card is the
right value target. If it approaches 32 GiB or concurrent calls are required,
use 48 GB.

## Operational Notes

Use on-demand instances for the inbound phone service. Interruptible instances
can pause without notice, and serverless cold starts are a poor fit for a model
stack of this size. Keep a test instance running until the WhatsApp call is
confirmed, then destroy it promptly so billing stops.
