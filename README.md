# Nano Flash / Hybrid TTS

Nano Flash is a research-stage hybrid text-to-speech architecture for low-latency, multilingual voice agents. The design combines autoregressive (AR) generation for fast speech startup, adaptive block-diffusion generation for efficient buffered continuation, and a one-step causal acoustic renderer.

> Status: architecture and product requirements specification. Nano Flash has not yet been trained or benchmarked, and the targets below require experimental validation.

## Core idea

Realtime voice agents need both a low time to first audio and enough sustained throughput to avoid playback underruns. Nano Flash assigns each decoding approach to the phase where it is strongest:

1. Begin with AR decoding to emit the first semantic frames immediately.
2. Render and stream the first 80–160 ms of audio with a one-step acoustic renderer.
3. Switch to block diffusion when committed text and playback-buffer headroom are sufficient.
4. Fall back to AR whenever text availability or confidence is low.

The AR and block-diffusion paths share the text conditioning, semantic vocabulary, embeddings, most Transformer blocks, and speaker/style conditioning to reduce mode-switch mismatch.

## Proposed architecture

```text
Incremental LLM text
        |
Streaming normalization and language detection
        |
Phoneme/subword and control tokens
        |
Incremental text encoder and commitment state
        |
Shared semantic backbone (100–130M)
        |-------------------------------|
        v                               v
AR startup head                  Block-diffusion head
        |                               |
        +---------------+---------------+
                        |
Semantic, duration, and prosody states
                        |
One-step causal acoustic renderer (30–60M)
                        |
Lightweight causal vocoder / codec decoder
                        |
Packetization, overlap/crossfade, playback buffer
                        |
Streaming PCM
```

## Initial product targets

| Dimension | Nano Flash v0.1 target |
| --- | --- |
| Primary use | Realtime multilingual conversational agents |
| Languages | English, Malayalam, and Hindi |
| Core parameter budget | 150–220M, excluding a separately counted vocoder |
| First semantic generation | Autoregressive |
| Steady-state generation | Adaptive block diffusion with AR fallback |
| Acoustic renderer | One-step distilled causal latent/mel renderer |
| Audio output | 24 kHz mono; optional 48 kHz later |
| Warm TTFA | `<150 ms` p50 and `<250 ms` p95 on a strong GPU |
| Realtime factor | `<0.20` at concurrency 1 on the target GPU |
| Controls | Speaker cloning, pace, pause, emotion/style, and non-verbal events |

## Functional scope

- Accept incrementally arriving UTF-8 text and track a committed-text boundary.
- Emit playable PCM before the complete utterance is available.
- Support zero-shot speaker conditioning from a short reference clip.
- Provide AR-only, block-only, and hybrid decoding modes.
- Support emotion, pace, pause, emphasis, and events such as laughter, sighs, and breaths.
- Cancel within one output packet and clear future queued audio.
- Report timestamps, token counts, TTFA, RTF, and per-stage profiling.
- Target a GPU server runtime first, followed by a distilled ONNX/CPU path.

## Development roadmap

The recommended sequence deliberately stabilizes one interface at a time:

1. Build a reproducible profiling harness for MOSS-TTS-Nano, Chatterbox-Nano, and Chatterbox-Flash.
2. Train a 100–130M multilingual pure-AR semantic baseline.
3. Add explicit prosody and paralinguistic controls.
4. Train a high-quality multi-step acoustic teacher.
5. Distill a causal one-step acoustic renderer.
6. Train the block-diffusion continuation head.
7. Train and evaluate mixed-mode switching with randomized switch points.
8. Optimize serving, batching, quantization, and the later CPU/edge path.

## Evaluation priorities

The main acceptance areas are:

- latency: cold start, warm TTFA, inter-chunk gaps, RTF, and cancellation latency;
- streaming stability: underruns across incoming text rates and concurrency levels;
- intelligibility: WER/CER, named entities, code-switching, numbers, and abbreviations;
- voice quality: speaker similarity, naturalness, prosody, and emotion adherence;
- hybrid continuity: human preference and artifact detection at AR-to-block boundaries;
- robustness: repetition, skipping, hallucination, long-form output, and noisy references.

## Repository contents

- [`docs/Nano_Flash_Technical_PRD.docx`](docs/Nano_Flash_Technical_PRD.docx) — full technical architecture and product requirements document.
- `README.md` — project overview and implementation direction.

## Influences

The design is informed by public work on [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano), [Chatterbox](https://github.com/resemble-ai/chatterbox), and [Chatterbox-Flash](https://github.com/resemble-ai/chatterbox-flash). Nano Flash is intended as an original implementation and does not assume checkpoint compatibility with those projects.

## Licensing and data

No project license has been selected yet. Before distributing code or weights, define the code/weight licensing strategy and maintain an auditable bill of materials for datasets, checkpoints, synthetic teachers, and speaker consent. The PRD treats training-data provenance and voice authorization as first-class requirements.
