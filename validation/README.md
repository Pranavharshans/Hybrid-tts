# Nano Flash validation harness

This directory contains the reproducible programs used by the experiment gates recorded in [`exp.md`](../exp.md). Large datasets, caches, generated audio, checkpoints, and raw run artifacts stay outside Git. Compact configurations, code, summaries, and decisions are version controlled.

## Operating rules

1. Run one controlled experiment at a time on the 16 GB GPU.
2. Record the exact goal and acceptance criteria before a run.
3. Use pretrained/frozen components before training new parameters.
4. Stop on non-finite loss, unrecoverable checkpoint failure, or less than 35 GB free disk.
5. Keep only the best, last, and one recovery checkpoint per experiment.
6. Push code and `exp.md` after every completed experiment.
7. Treat automatic metrics as architectural evidence, not a substitute for final human listening.

## G0 environment check

```bash
python validation/g0_environment.py \
  --workdir /workspace/nano-flash-artifacts/g0 \
  --output /workspace/nano-flash-artifacts/g0/environment.json
```

The command verifies CUDA execution, real filesystem capacity, sequential temporary-file I/O, atomic checkpoint replacement, and checkpoint reload equality. It exits nonzero when a critical condition fails.

`g0_supervisor_probe.sh` is a five-second one-shot workload used to prove that the instance supervisor can own a validation process independently of the initiating SSH connection and atomically publish its completion record.
The matching `supervisor/nano-flash-probe.conf` is copied into the instance supervisor configuration only for EXP-003.

## G1 MOSS smoke inference

`moss_smoke.py` runs a deterministic English voice-clone request on CUDA, validates the generated WAV structurally, and records cold elapsed time plus peak allocated VRAM. `run_moss_smoke.sh` and `supervisor/moss-smoke.conf` make the potentially download-heavy first run independently supervised.

The current Blackwell image ships PyTorch 2.12 without a matching torchaudio wheel. MOSS requires only audio load, save, and resampling, so `compat/torchaudio` provides that narrow API with SoundFile and SciPy while preserving the validated CUDA framework instead of installing an ABI-mismatched torchaudio build.

## G1 MOSS warm profile

`moss_profile.py` loads exact local Hugging Face model, nested text-tokenizer, and audio-tokenizer snapshots once in offline mode, primes the model and codec, then runs three uninstrumented deterministic requests for warm TTFA, RTF, chunk timing, VRAM, and output repeatability. A fourth synchronized run separately attributes wall time to prompt encoding, semantic generation, codec decoding, and orchestration/I/O. Product-target comparisons are reported as baseline facts; they are not part of the profiling experiment's pass condition.

`run_moss_profile.sh` contains the immutable snapshot paths used on the validation instance. `supervisor/moss-profile.conf` keeps the run alive across SSH disconnects.

## G1 Chatterbox-Flash environment

`setup_chatterbox_flash.sh` creates a separate Python 3.12 environment for the exact official Chatterbox and Chatterbox-Flash source revisions. It installs the upstream-supported PyTorch 2.7.1 CUDA 12.8 ABI family, validates a real Blackwell BF16 kernel, and writes an atomic environment record. This isolation is mandatory: Chatterbox's metadata pins an older Torch/torchaudio pair and must not alter the working MOSS PyTorch 2.12 CUDA 13.0 environment.

The first reproducible backend is pure Torch SDPA. FlashInfer remains a later optimization experiment because its compiled extension adds an independent ABI and GPU-architecture risk.

`setup_chatterbox_nano.sh` separately recreates the official Nano demo Space's PyTorch 2.11 and Transformers 4.46 framework family with CUDA 13.0 Blackwell wheels. It imports the Nano-only loader from the exact pinned official Space source and consumes the gated checkpoint only through its already downloaded immutable local snapshot. No Hugging Face credential is needed or stored for later offline runs.

`chatterbox_nano_profile.py` loads that snapshot offline, prepares and caches one zero-shot speaker reference, runs a warm-up, three fixed-seed uninstrumented measurements, and one instrumented stage pass. Because the released Nano API returns only a completed utterance, its measured time-to-first-audio equals full generation time; the harness records that limitation explicitly instead of presenting completion latency as streaming TTFA.
