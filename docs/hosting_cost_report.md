# Voice Runtime Hosting Cost Report

Snapshot date: 2026-07-29

## Recommendation

Run one always-on Vast.ai RTX 3060 12 GB instance with a 32 GB disk. Use the
checked-in CUDA profile and keep all 42 Gemma layers on the GPU.

- Absolute lowest observed listing: about **$0.03956/hour** or **$28.88/month**
  including 32 GB storage. This host was in China, so verify access to Meta and
  Hugging Face before relying on it.
- Lower-risk observed listing: about **$0.05393/hour** or **$39.37/month**
  including 32 GB storage for a verified RTX 3060 host in Romania.
- The US benchmark contract cost **$0.06148/hour**, or **$44.88/month** at 730
  hours.

Vast prices and availability change continuously. The totals above add the
listing's compute price to its per-GB storage price for 32 GB. Network transfer
is usage-based and is not included.

## Measured Runtime

The production stack was tested end to end on an RTX 3060 with 12,288 MiB VRAM
and eight effective CPU cores:

| Measurement | Result |
| --- | ---: |
| Full-stack peak VRAM | 6,965 MiB |
| Warm Whisper load | 28.8 s |
| Whisper on 13.4 s reference audio | 9.0 s |
| Warm Gemma load, 42 GPU layers | 2.5 s |
| Gemma sample response | 2.8 s |
| Warm OmniVoice V2 load | 13.3 s |
| OmniVoice V2, 5.1 s generated audio | 3.3 s |
| TTS real-time factor | 0.66 |
| Installed environment | 15 GB |
| Hugging Face model cache | 9.3 GB |
| Free space on 32 GB disk | 8.4 GB |

The loaded TTS snapshot was
`2broke2code/serendib-omnivoice-finetuned-v2` revision
`b0cf77a8ad8a50881d5fa992aee714d626e97c7b`. A synthesized Sinhala sentence
was transcribed back successfully by the production ASR model.

The same test with only 24 Gemma GPU layers took 12.7 seconds for the sample
response. Full offload reduced that to 2.8 seconds while increasing peak VRAM
by less than 1 GB, so CPU offload is not appropriate for this workload.

## Why Not 8 GB

The measured stack could fit narrowly on an 8 GB Ampere GPU, but it would leave
less than 1 GB of VRAM headroom. The cheapest reliable 8 GB offers also had
roughly the same hourly compute price as the 12 GB tier and substantially less
system RAM. That creates OOM and concurrency risk without meaningful savings.

## Provider Comparison

Observed or published starting prices at the snapshot date:

| Provider | Suitable tier | Approximate monthly cost |
| --- | --- | ---: |
| Vast.ai | RTX 3060 12 GB, lowest observed plus 32 GB disk | $28.88 |
| Vast.ai | RTX 3060 12 GB, lower-risk observed plus 32 GB disk | $39.37 |
| Salad | RTX 3060 12 GB batch price | $61.32 |
| RunPod | RTX A5000 24 GB | $197.10 |

Vast.ai is market-priced and bills compute, storage, and bandwidth separately:
<https://docs.vast.ai/guides/instances/pricing>. Salad publishes its container
prices at <https://salad.com/pricing>, and RunPod publishes its GPU prices at
<https://www.runpod.io/pricing>.

Interruptible instances are inappropriate for an inbound phone service because
they can pause at any time. Serverless GPU cold starts are also a poor fit for
this 24 GB on-disk stack unless an always-on front end and long first-call delay
are acceptable.

