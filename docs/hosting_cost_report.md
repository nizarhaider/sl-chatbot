# Voice Runtime Hosting Cost Report

Snapshot date: 2026-08-16

## Recommendation

Use two distinct sizing rules for the voice agent:

- **24 GB VRAM is the hard runtime floor.** Select the cheapest compatible,
  verified listing at or above this capacity.
- **48 GB VRAM is the training and validation tier.** Use it for LoRA training,
  adapter merging, GGUF conversion, and the first end-to-end deployment. It is
  not automatically required for steady-state hosting.

The deployer rejects configurations below 24 GB.

## Cheapest Live Vast.ai Listings

The deployer's compatibility and reliability query was run against live
on-demand listings with an 80 GB disk on 2026-08-16:

| Tier | Cheapest observed GPU | Hourly total | 730-hour month |
| --- | --- | ---: | ---: |
| 24 GB+ | Quadro RTX 8000 48 GB | $0.25111 | $183.31 |
| 48 GB+ | Quadro RTX 8000 48 GB | $0.25111 | $183.31 |

There was no cheaper compatible 24 GB listing at the snapshot, so the 24 GB
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

The production host must provide at least 24 GB of VRAM. Use 48 GB when the
live workload requires more headroom or concurrent calls.

## Operational Notes

Use on-demand instances for the inbound phone service. Interruptible instances
can pause without notice, and serverless cold starts are a poor fit for a model
stack of this size. Keep a test instance running until the WhatsApp call is
confirmed, then destroy it promptly so billing stops.
