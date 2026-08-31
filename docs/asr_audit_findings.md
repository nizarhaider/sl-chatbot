# Archived-call ASR audit — 2026-09-01

This audit ran the production checkpoint, `SPEAK-ASR/whisper-medium-si-merged`
(Hub revision `9fbeaaf862c4befbcf9481338004255a8576e888`), over the caller channel of
every archived call available on the audit date. It produced 53 non-empty caller
turns, totalling 120.6 seconds. Raw recordings and full caller transcripts are
intentionally excluded from this repository.

## Findings

The sample is too small, and lacks human reference transcripts, to report WER or
claim a confirmed error count. It does, however, give clear, high-priority review
targets for the real-estate flow:

| Target | Production output pattern | Assessment | Training implication |
| --- | --- | --- | --- |
| `නුගේගොඩ` (Nugegoda) | A multi-token phonetic near-miss rather than the canonical place name | High-confidence location normalisation failure | Record standalone and conversational variants; include `නුගේගොඩේ`, `නුගේගොඩට`, and English-influenced pronunciation. |
| `රාජගිරිය` (Rajagiriya) | Voiced `ග` was rendered as `ක`, with an inflected ending | High-confidence consonant/place-form failure | Pair nominative and postposition forms such as `රාජගිරියේ`. |
| `කොළඹ` (Colombo) | English-influenced `කළම්බෝ` variants appeared instead of canonical Sinhala spelling, while another call was correct | High-confidence canonicalisation inconsistency | Include Sinhala and code-switched `Colombo` pronunciations with one canonical label. |
| `අංගොඩ` (Angoda) | A severely distorted multi-word output resembles this place request | Candidate; confirm against the clipped caller audio before training | Add only after reviewer verification, then collect nearby-place contrast pairs. |

Number and appointment language was less uniformly wrong but structurally fragile:

- `තුන` and an English `50 million` amount were sometimes recognised, but adjacent
  property words were corrupted.
- A date/time request retained isolated `3` and `6` tokens while duplicating or
  losing surrounding temporal context.
- Training labels must preserve the spoken form and a separate normalized semantic
  field; do not replace `හවස තුනට` with `15:00` in ASR targets.

## Required review before fine-tuning

The audit runner emits one caller-turn clip boundary and production transcript.
For every candidate above, a Sinhala reviewer should mark `reference_text`,
`place_entities`, `number_entities`, and `confirmed_error_type`. This makes the
next evaluation set speaker-disjoint and keeps uncertain automatic transcripts out
of supervised training.

## Corpus admission rules

`scripts/curate_asr_manifest.py` enforces the operational rules:

1. Reject OpenSLR-52 and aliases, which are a known widespread Sinhala ASR source.
2. Require a reusable rights basis and a URL proving it.
3. Require public material to postdate 2026-05-05, the deployed model's Hub
   creation date, or require owner permission plus proof that the clip was
   previously unpublished.
4. Reject exact duplicate audio and duplicate normalized transcripts across sources.

The checkpoint does not disclose a complete training-data manifest. Therefore,
content-level non-overlap with undisclosed upstream data cannot be proved; the
publication-date/private-origin rule is the strongest auditable exclusion boundary
available without that manifest.
