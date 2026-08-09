# SerendibAI Voice Model Report

Last updated: 2026-08-09

This is the single canonical operational report for OmniVoice training,
evaluation, release tracking, and the WhatsApp voice runtime. Historical V2
runtime and hosting measurements are retained here only where they remain
useful for comparison.

## Current Status

- Runtime target: `2broke2code/serendib-omnivoice-finetuned-v4`.
- Immutable model revision: `85747b9376885e3bf8847f6dfd45864798e31ccd`.
- Release tag: `v4.0.0`.
- Release training checkpoint: step 1,104, predetermined before training.
- Runtime reference: training row `033`, stored as
  `app/voices/training-033.wav`, SHA-256
  `dcf030654b31bd745280a2d8d942ac47e79bdf7cd2aca3e1b047a03d3e810fc5`.
- V4 manual holdout IDs 310 and 311 were generated on the training GPU and
  packaged with their real recordings for listening comparison.

## System Architecture

The production call path is entirely local on the GPU host:

```text
WhatsApp Cloud webhook
  -> FastAPI /webhook
  -> WhatsApp Calling SDP offer
  -> aiortc RTCPeerConnection
  -> inbound WhatsApp Opus audio
  -> local RMS VAD
  -> SPEAK-ASR/whisper-medium-si-merged
  -> Gemma 4 E4B Q4 through llama.cpp
  -> RealtimeTTS with SerendibAI OmniVoice V4
  -> outbound 48 kHz stereo PCM
  -> WhatsApp call audio
```

The assistant brain remains local Gemma. No hosted LLM API is in the voice
path. Neon stores property data, appointments, call state, and transcripts.

## V4 Release

V4 is separate from V3 and starts from the same pinned base model. The private
dataset contains all 311 normalized recordings, but only IDs 001 through 309
enter tokenization and gradient training. IDs 310 and 311 are isolated as the
manual holdout and were physically absent from the GPU until the training
process exited.

```text
dataset: 2broke2code/serendib-omnivoice-dataset-v4@b8424aeaa0687bae2fca6b1feb723cefc406721a
model: 2broke2code/serendib-omnivoice-finetuned-v4@85747b9376885e3bf8847f6dfd45864798e31ccd
tag: v4.0.0
dataset manifest: f96d32b86ca07b02eb363be8c2fc92c30a89ffd6ead370f2fcd51e7213ea18be
model SHA-256: ef1ba42bfed2a341312b558f19c288f496be85b5f62a5d02a31c755845b4342a
```

The RTX 5060 Ti run completed 1,104 steps in 964 seconds. It used bf16/SDPA,
learning rate `1e-5`, a cosine schedule, 3% warmup, batch tokens 512, gradient
accumulation 8, and seed 42. There was no validation split and no
validation-loss checkpoint selection; the predetermined final checkpoint was
published. Vast.ai instance `47170301` ran for 2,441.738 seconds at
`$0.0911111111` per hour, for an estimated `$0.0618`, and was destroyed after
the release was verified.

The original V4 release check generated valid 24 kHz mono PCM for both unseen
manual-test scripts using `app/voices/female-004.wav`. The live runtime now uses
training row `033` instead:

| ID | Real duration | V4 duration | V4 SHA-256 |
| ---: | ---: | ---: | --- |
| 310 | 11.113 s | 9.760 s | `d5553feef07b73f228e4194b8bd180a8b05e8f44d5834fd8413b08ece71b2c7b` |
| 311 | 7.700 s | 8.220 s | `4f879eb3388e084baab436370e6e9c587207e6af2b4581ca9e729dd041c3f33c` |

These format, level, and hash checks establish artifact integrity. Perceptual
quality remains a manual listening decision; the private model release
contains the real/generated pairs.

## V3 Recording Intake

- Source: `Serendib AI Voice Model Training` WhatsApp group.
- New recording IDs: `223` through `311`.
- New recordings: 89, with no missing IDs.
- Download size: 15,007,207 bytes.
- Source format: AAC in M4A/MP4 containers, 48 kHz stereo.
- Normalized format: 24 kHz mono 16-bit PCM WAV.
- New-audio duration: 896.927 seconds (14.949 minutes).
- Duration range: 7.529 to 13.247 seconds.
- Quality checks: no silence, clipping, duration, or duplicate-hash flags.

## V3 Dataset

The cumulative V3 dataset contains 311 recordings totaling 3,351.970 seconds
(55.866 minutes).

| Split | Recordings | Rule |
| --- | ---: | --- |
| Train | 280 | All remaining IDs |
| Validation | 30 | IDs divisible by 10, except ID 200 |
| Test | 1 | ID 200, preserved from V2 for comparison |

The split extends the V2 rule without moving an existing recording between
train, validation, and test.

Dataset manifest SHA-256:

```text
cd847634bd70547645cc0cd75a868ae87242ffea0008177a324bea1a9ec6c5d9
```

Private dataset release:

```text
2broke2code/serendib-omnivoice-dataset-v3@0eefb4f68e7c2e05016a87ac8f24f3ad0d282a8c
tag: v3.0.0
```

Private model release:

```text
2broke2code/serendib-omnivoice-finetuned-v3@5857eb287f856364ce8c2440c8043cc42b1de791
tag: v3.0.0
```

## Training Configuration

- Pipeline: official OmniVoice audio tokenization, then single-GPU fine-tuning.
- OmniVoice source revision: `468e927ba3716cd8dd86421148dfb3046e9f9d7b`.
- Base model: `k2-fsa/OmniVoice` revision
  `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`.
- Steps: 1,000.
- Precision and attention: bf16 with SDPA.
- Learning rate: `1e-5`, cosine schedule, 3% warmup.
- Batch tokens: 512.
- Gradient accumulation: 8.
- Seed: 42.
- Validation and checkpoint interval: 250 steps.
- Selection rule: lowest validation loss, not the final checkpoint.

## Training Results

All 1,000 steps completed on Vast.ai instance `47051449`, an RTX 3090 with
24 GB VRAM. Model training took 1,900 seconds. The complete rental lasted
3,918 seconds, including setup, tokenization, evaluation, and release upload.

| Step | Validation loss | Selection |
| ---: | ---: | --- |
| 250 | **3.4741** | Best; published |
| 500 | 3.7607 | Not selected |
| 750 | 3.4781 | Not selected |
| 1,000 | 3.5289 | Not selected |

The published model blob is 2,450,344,144 bytes. SHA-256:

```text
5bedf7d6d4bf52c8cf61ed09282699246779a574586bf8ba5dab7acc8e914ddd
```

The V3 validation set contains nine recordings absent from V2, so their
validation losses are not a strict apples-to-apples comparison.

## Evaluation

### Held-out regression sample

Checkpoint 250 generated a deterministic sample for held-out script 200 using
reference recording 033 and seed 42. The output is 9.400 seconds of 24 kHz
mono 16-bit PCM WAV.

| Artifact | SHA-256 |
| --- | --- |
| Generated V3 sample | `4aa2378121e5f714e57fc184f61a6271b241a37192bb48b153430aa10ec0b3c5` |
| Reference 033 | `dcf030654b31bd745280a2d8d942ac47e79bdf7cd2aca3e1b047a03d3e810fc5` |
| Held-out recording 200 | `283a84d5e75745a9b574f3b72809a2df59fcc1352f5efb99cb6ffb1d75e78d1a` |

The reference and held-out hashes exactly match the V2 comparison assets.

### Unseen Base-versus-V3 comparison

Scripts 312, 330, and 369 do not appear in the 1-311 V3 training corpus. Each
was synthesized locally with the pinned base model and pinned V3 model, using
the production reference `app/voices/female-004.wav` and the same deterministic
seed for both variants.

- Runtime: Apple MPS, float32.
- Scope: OmniVoice only; no ASR, Gemma, webhook, or server was started.
- Format: 24 kHz mono 16-bit PCM WAV.
- Integrity: six unique hashes, nonzero RMS, and no empty output.

| Script | Variant | Seed | Duration | RMS | SHA-256 |
| ---: | --- | ---: | ---: | ---: | --- |
| 312 | Base | 20260816 | 8.62 s | 0.037294 | `01c97e8fec7fb716d278588d53952e979e2088ecbd9049ea9b7ff1e868ed771e` |
| 312 | V3 | 20260816 | 7.98 s | 0.033538 | `a0dddfa7dabc115ad77f7ee198d9969c601efc1d31392d0ffb03738d45cdaa79` |
| 330 | Base | 20260834 | 8.22 s | 0.038523 | `3567a6cffd01f202a0a316da87cd044539e9291a50487290bce885c2bf510a94` |
| 330 | V3 | 20260834 | 7.67 s | 0.036467 | `b1223d921c22a66f0554febe202b433c31a9a570b62db0e45e38e3b0c9dca0d7` |
| 369 | Base | 20261096 | 7.63 s | 0.034086 | `3668ef61978998549d22a811e0722d5a2066dee6a03b79ead95cc2ee84bc0428` |
| 369 | V3 | 20261096 | 6.50 s | 0.030233 | `4f9fe737599dadd5966a87e2e00f0ccd34e907315a687a71cb16ac81de11a84c` |

The test report was delivered to the `SerendibAI` WhatsApp group. The six WAV
attachments remain pending because the Chrome extension currently lacks file
URL access; the pre-existing WhatsApp draft was preserved.

## Tracking and Storage

| Item | Location |
| --- | --- |
| Dataset and normalized audio | Private Hugging Face dataset repo `2broke2code/serendib-omnivoice-dataset-v4` |
| Model, configs, logs, metrics, and evaluation WAVs | Private Hugging Face model repo `2broke2code/serendib-omnivoice-finetuned-v4` |
| Reproducible configs and notebook records | `serendibai-omnivoice-finetuning` sub-repository |
| Runtime configuration and this report | `sl_chatbot` sub-repository |
| Experiment tracker | Local MLflow run `58a9adcfbf6848a7b3d0537cf87bd3b5` (`FINISHED`) |
| Local V4 manual-test WAVs and manifest | `sl_chatbot/comparison_v4/` |

Large datasets, weights, checkpoints, and generated artifacts are stored on
the private Hugging Face Hub rather than in the local workspace.

## Cost and Capacity

V4 training and release preparation cost an estimated `$0.0618` at
`$0.0911111111` per hour. The instance was destroyed and the active-instance
count was verified as zero.

V3 training cost `$0.1548` at `$0.1422222222` per hour. The training instance
was destroyed after the release was verified.

Historical production measurements showed the full stack peaking at 6,965 MiB
on an RTX 3060 12 GB, with 15 GB installed environment, 9.3 GB Hugging Face
cache, and 8.4 GB free on a 32 GB disk. A prior RTX 5080 16 GB runtime used
about 13.1 GiB after loading a different environment snapshot. Host, package,
and cache differences explain the spread; at least 16 GB VRAM and a 40 GB disk
remain the safer deployment target.

Vast.ai pricing changes continuously. Historical viable offers including disk
ranged from roughly $29 to $45 per month for always-on RTX 3060-class hosts.
Interruptible instances are unsuitable for an inbound phone service.

## Historical Runtime Baseline

The June 2026 end-to-end call test completed 11 turns without application,
empty-response, empty-transcript, or TTS failures.

| Metric | Average | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| Whisper STT | 321 ms | 135 ms | 549 ms |
| Gemma LLM | 592 ms | 277 ms | 1,071 ms |
| TTS wall time | 946 ms | 381 ms | 1,844 ms |
| Synthesized audio | 9,389 ms | 2,720 ms | 18,400 ms |
| Post-VAD response | 1,860 ms | 817 ms | 3,190 ms |
| Total turn | 4,822 ms | 2,714 ms | 8,311 ms |

Sinhala ASR quality for noisy and code-mixed speech was the main functional
bottleneck. Gemma and TTS were reliable, but overly long responses sometimes
hurt call pacing.

## Progress Log

| Time (+0530) | Result |
| --- | --- |
| 2026-08-07 11:39 | Downloaded all 89 new WhatsApp recordings. |
| 2026-08-07 11:44 | Verified IDs 223-311, AAC codec, and source file integrity. |
| 2026-08-07 11:49 | Normalized new audio and built cumulative V3 manifests. |
| 2026-08-07 11:51 | Added reproducible V3 training and data configs. |
| 2026-08-07 11:53 | Published and tagged the immutable private V3 dataset. |
| 2026-08-07 11:56 | Rented RTX 3090 Vast instance `47051449`. |
| 2026-08-07 12:09 | First tokenization attempt stopped because OpenBLAS expanded to 64 threads per worker; no training began. |
| 2026-08-07 12:12 | Restarted with BLAS/OMP thread pools pinned to one. |
| 2026-08-07 12:17 | Tokenized all 280 training and 30 validation samples with zero failures. |
| 2026-08-07 12:25 | Saved checkpoint 250; validation loss `3.4741`. |
| 2026-08-07 12:33 | Saved checkpoint 500; validation loss `3.7607`. |
| 2026-08-07 12:41 | Saved checkpoint 750; validation loss `3.4781`. |
| 2026-08-07 12:49 | Completed 1,000 steps; final validation loss `3.5289`. |
| 2026-08-07 12:50 | Selected checkpoint 250 and generated held-out evaluation audio. |
| 2026-08-07 12:57 | Published and tagged V3; verified all 21 Hub files and model SHA-256. |
| 2026-08-07 12:59 | Destroyed training instance; final estimated cost `$0.1548`. |
| 2026-08-07 13:05 | Sent the first completion report to the user's WhatsApp self-chat. |
| 2026-08-07 14:23 | Destroyed Vast.ai instance `47058959` and verified zero active instances. |
| 2026-08-07 14:33 | Completed three base-versus-V3 comparisons locally on Apple MPS. |
| 2026-08-07 14:35 | Delivered the local comparison report to the `SerendibAI` WhatsApp group. |
| 2026-08-07 14:36 | WAV upload paused because Chrome extension file URL access is disabled. |
| 2026-08-08 17:03 | Rented verified RTX 5060 Ti Vast.ai instance `47170301` at `$0.0911111111` per hour. |
| 2026-08-08 17:13 | Tokenized IDs 001-309 with zero failures; IDs 310-311 remained absent. |
| 2026-08-08 17:32 | Completed the predetermined 1,104-step V4 final checkpoint. |
| 2026-08-08 17:34 | Generated and verified the real-versus-V4 manual holdout pairs for IDs 310 and 311. |
| 2026-08-08 17:42 | Published and tagged private V4 model revision `85747b9376885e3bf8847f6dfd45864798e31ccd`. |
| 2026-08-08 17:44 | Destroyed instance `47170301`; verified zero active Vast.ai instances. |

## Operations

Healthy startup and call logs should show:

```text
WEBHOOK_VERIFIED
Received call event: connect
Received audio track from WhatsApp
Connection state: connected
Turn VAD: Speech started
Turn VAD: Speech ended
Turn transcript
Turn response
RealtimeTTS complete
```

Useful checks when a Vast host is deliberately deployed:

```bash
tail -f /workspace/sl-chatbot/run_logs/important.log
tail -f /workspace/sl-chatbot/run_logs/webhook.log
nvidia-smi
tmux ls
curl -sS http://127.0.0.1:8081/
```

There is currently no active Vast host. Always verify the active-instance count
after testing and destroy any temporary rental promptly.
