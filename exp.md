# Nano Flash Lean Validation Ledger

This file is the authoritative experiment record for the unattended English-only Nano Flash validation program. Every experiment records its goal, configuration, evidence, result, and architectural decision. Code changes and ledger updates are committed and pushed after each completed experiment.

## Final decision contract

The program ends with one of three verdicts:

- **YES** — the complete hybrid architecture is technically validated at sanity scale.
- **CONDITIONAL YES** — the core streaming product works, but one or more proposed components should be removed or redesigned.
- **NO** — evidence does not justify scaling the architecture in its current form.

The technical validation score is evidence coverage, not a probability of commercial success.

| Gate | Weight | Validates |
| --- | ---: | --- |
| G0 — Environment and reproducibility | 5% | Reliable unattended execution and recovery |
| G1 — Pretrained baseline profiling | 10% | Actual latency and quality bottlenecks |
| G2 — Incremental-text behavior | 10% | Safe partial-text streaming behavior |
| G3 — Tiny overfit | 15% | Training pipeline, representation, and conditioning correctness |
| G4 — Small English AR | 20% | Held-out intelligibility and semantic stability |
| G5 — Streaming runtime | 15% | TTFA, RTF, gaps, buffering, and cancellation |
| G6 — One-step renderer | 10% | Acoustic latency reduction without unacceptable degradation |
| G7 — Block continuation | 10% | Material throughput gain over cached AR |
| G8 — Integrated hybrid | 5% | Safe AR/block switching and fallback |

Critical gates are G3, G4, and G5. A full-hybrid YES additionally requires G7 and G8 to pass. G6 or G7 may fail while the result remains a CONDITIONAL YES if the core streaming architecture passes.

## Resource envelope

- GPU: NVIDIA GPU with 15.9 GiB usable VRAM (Vast instance 45075107)
- CPU: 24 vCPU, Xeon E5-2643 v4
- RAM: 128.6 GiB
- Disk budget: 300 GB persistent instance disk
- Network: approximately 825 Mbps down / 877 Mbps up
- Strategy: reuse pretrained components, cache expensive representations, train adapters/small heads, retain only best/last/recovery checkpoints
- Disk safety floor: stop new writes below 35 GB free

## Experiment record template

Each experiment uses this structure:

```text
### EXP-XXX — Title
Gate:
Status: PLANNED | RUNNING | PASS | FAIL | INCONCLUSIVE | BLOCKED
Started:
Finished:
Commit:

Goal:
Configuration:
Acceptance criteria:
Commands/artifacts:
Results:
Decision:
Follow-up:
```

## Experiment index

| ID | Gate | Experiment | Status | Result |
| --- | --- | --- | --- | --- |
| EXP-000 | G0 | Local repository and ledger initialization | PASS | Repository clean; ledger established |
| EXP-001 | G0 | Remote GPU discovery and SSH connectivity | PASS | Running instance; noninteractive SSH authenticated |
| EXP-002 | G0 | GPU/CUDA/PyTorch capability validation | PASS | Blackwell-capable cu130 PyTorch; FP16/BF16 CUDA operations valid |
| EXP-003 | G0 | Disk, network, checkpoint, and recovery validation | PASS | 300 GiB available; atomic checkpoints and detached supervisor verified |
| EXP-010 | G1 | MOSS-TTS-Nano installation and smoke inference | PASS | Deterministic CUDA inference produced valid 7.68 s English audio |
| EXP-011 | G1 | MOSS latency and resource profile | PASS | Warm TTFA passes target; RTF 1.13 exposes semantic-generation bottleneck |
| EXP-012 | G1 | Chatterbox access and isolated environment validation | PASS | Flash CUDA/ABI stack valid; gated Nano revision downloaded without persisting credentials |
| EXP-013 | G1 | Chatterbox-Nano smoke and warm profile | PASS | 0.265 warm RTF and 2.79 GiB peak; full-utterance API misses interactive TTFA |
| EXP-014 | G1 | Chatterbox-Flash smoke and warm profile | PASS | Torch baseline is deterministic and memory-light; 0.592 RTF exposes unoptimized semantic bottleneck |
| EXP-020 | G2 | Incremental-text simulator and prompt suite | PASS | 250 randomized chunkings preserve monotonic, lossless committed prefixes |
| EXP-021 | G2 | Partial-text stability across arrival rates | PASS | Stable at all rates; two-word safety alone cannot meet the TTFA target |
| EXP-030 | G3 | Token/data pipeline integrity | PASS | Official codec and SFT packer produce deterministic valid 100-record corpus |
| EXP-031 | G3 | 100-sample deliberate overfit | PASS | Loss fell 90.4% in 40 steps; finite reloadable checkpoint produced |
| EXP-032 | G3 | Checkpoint resume and reproducibility | PASS | Restored step exactly matches uninterrupted loss, model, and optimizer hashes |
| EXP-040 | G4 | Lean English AR adaptation | PASS | Stable 100-step adaptation improves disjoint-speaker held-out loss by 0.424% |
| EXP-041 | G4 | Held-out intelligibility and failure analysis | FAIL | Adapted WER regresses 19% to 32% through clause/utterance truncation |
| EXP-050 | G5 | Streaming scheduler and packetizer | PASS | Seven tests validate lossless packets, routing, buffer safety, fallback, and cancellation |
| EXP-051 | G5 | TTFA/RTF/gap/cancellation stress matrix | PASS | Measured stack has no feasible case; optimized AR 0.75/block 0.20 RTF passes all text-sufficient cases |
| EXP-060 | G6 | Renderer target caching | PASS | Tokens, conditioning, and RNG reproduce ten waveforms exactly at renderer RTF 0.038 |
| EXP-061 | G6 | One-step/two-step renderer feasibility | PASS | One step is RTF 0.024 but diverges strongly; retain already-fast two-step default |
| EXP-069 | G7 | FlashInfer Blackwell compatibility | PASS | Locked backend preserves ABI and executes fused CUDA kernel on Blackwell |
| EXP-070 | G7 | Frozen-backbone block head training | PLANNED | — |
| EXP-071 | G7 | AR versus block throughput/quality comparison | PLANNED | — |
| EXP-080 | G8 | Integrated AR/block switching | PLANNED | — |
| EXP-081 | G8 | Final adversarial streaming matrix | PLANNED | — |

## Records

### EXP-000 — Local repository and ledger initialization

- **Gate:** G0
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Establish the authoritative, version-controlled experiment ledger before remote mutations or training begin.

**Configuration:** Existing `main` branch tracking `origin/main`; source PRD stored in `docs/`.

**Acceptance criteria:** Clean repository, reachable Git remote, explicit verdict contract, weighted gate definitions, resource envelope, and planned experiment index.

**Commands/artifacts:** `git status`, `git remote -v`, `git log`; this file.

**Results:** The repository was clean and synchronized with `origin/main`. The validation contract and initial experiment index are recorded here.

**Decision:** PASS. Proceed to remote GPU discovery and environment validation.

**Follow-up:** Complete EXP-001 and append the discovered SSH/runtime facts without recording credentials.

### EXP-001 — Remote GPU discovery and SSH connectivity

- **Gate:** G0
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Discover the active Vast instance endpoint through the provider API and prove unattended, noninteractive SSH access with the registered local identity.

**Configuration:** Vast instance `45075107`, machine `143755`, direct SSH transport, root container user, batch authentication, bounded connection timeout.

**Acceptance criteria:** Provider reports the requested instance running; an SSH URL can be derived; a locally registered identity matches the account; `BatchMode=yes` authentication succeeds without interactive input.

**Commands/artifacts:** Read-only Vast instance query and SSH URL query; remote `id`/hostname connectivity probe. Credentials, public endpoint, and API response tokens are intentionally excluded from Git.

**Results:** The provider reported the instance as running with the requested RTX 5060 Ti configuration. The account had a registered SSH public key matching an existing local private identity. Noninteractive authentication succeeded as `root`, and the remote container returned a stable hostname.

**Decision:** PASS. The instance can support unattended orchestration from this workspace.

**Follow-up:** Run GPU, CUDA, PyTorch, memory, storage, process-persistence, and checkpoint-integrity validation in EXP-002/003.

### EXP-002 — GPU/CUDA/PyTorch capability validation

- **Gate:** G0
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Prove that the installed framework stack can execute real CUDA kernels on the Blackwell GPU in both intended mixed-precision formats.

**Configuration:** RTX 5060 Ti, 16,311 MiB reported VRAM (15.478 GiB through PyTorch), driver `595.71.05`, compute capability `12.0`, PyTorch `2.12.0+cu130`, bundled CUDA `13.0` runtime.

**Acceptance criteria:** CUDA device available; framework wheel targets a CUDA version new enough for Blackwell; FP16 and BF16 matrix multiplication complete; outputs contain only finite values; core CUDA libraries are present.

**Commands/artifacts:** `nvidia-smi`; `vast-capabilities metrics,packages`; isolated PyTorch FP16 and BF16 CUDA matrix multiplications.

**Results:** The live manifest reported the complete required CUDA component set, a CUDA 13.0 PyTorch wheel, driver support through CUDA 13.2, and Blackwell compute capability 12.0. A 2048×2048 FP16 matmul completed in 0.1684 s and a 1024×1024 BF16 matmul completed in 0.1398 s; both outputs were finite. PyTorch reports native BF16 support. The first inline probe had a shell-quoting syntax error and performed no GPU assertion; the corrected isolated probes passed.

**Decision:** PASS. The installed framework is architecture-compatible and suitable for lean mixed-precision training.

**Follow-up:** Validate disk safety, persistent job supervision, checkpoint integrity, and off-instance recovery in EXP-003.

### EXP-003 — Disk, checkpoint, supervision, and recovery validation

- **Gate:** G0
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Prove that the real instance filesystem, checkpoint path, detached process manager, repository synchronization, and off-instance evidence path are safe enough for unattended validation.

**Configuration:** `/workspace` container storage with a 300 GiB allocation and no attached Vast volume; 35 GiB hard safety floor; 64 MiB fsync-backed I/O probe; atomic PyTorch checkpoint; one-shot Supervisor process; GitHub as code/ledger authority and SSH copy for compact raw evidence.

**Acceptance criteria:** At least 35 GiB free; nonzero practical read/write throughput; atomically saved checkpoint reloads with identical model tensors, optimizer state, and stable SHA-256; a Supervisor-owned workload completes after its initiating SSH session ends; code can synchronize from the public repository; evidence can be copied off the non-volume-backed instance.

**Commands/artifacts:** `validation/g0_environment.py`; `validation/g0_supervisor_probe.sh`; `validation/supervisor/nano-flash-probe.conf`; ignored raw artifacts `artifacts/g0/environment.json` and `artifacts/g0/supervisor-probe.json`.

**Results:** The filesystem reported 300.000 GiB total and 299.959 GiB free. The fsync-backed temporary write measured 419.674 MiB/s and the warm sequential read measured 2343.289 MiB/s. A real CUDA training step was saved atomically; model tensors matched exactly after reload, optimizer state was present, and the checkpoint digest remained stable. The Supervisor probe completed five seconds after launch and published its atomic completion record after the launching SSH session had closed. Compact evidence was copied off-instance and hashed locally (`31452c…20534` for the environment record and `fe5be7…5312` for the supervisor record). The instance manifest confirms `/workspace` is not a volume, so Git and off-instance copies remain mandatory. An initial anonymous clone failed while the repository was private; after the user made it public, a clean clone at the pushed revision succeeded. The probe also exposed a missing executable bit, which was corrected and pushed in commit `e42b1bd`.

**Decision:** PASS. G0 is complete; the environment can run recoverable unattended jobs within the defined storage policy.

**Follow-up:** Begin G1 by installing MOSS-TTS-Nano in an isolated project environment and running deterministic smoke inference before profiling.

### EXP-010 — MOSS-TTS-Nano installation and smoke inference

- **Gate:** G1
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Prove that the current upstream MOSS-TTS-Nano checkpoint can run deterministic English voice-clone inference on the validated Blackwell framework without downgrading CUDA/PyTorch.

**Configuration:** Upstream Git commit `11619374849c649486584e3b10ed55b176a924ee`; Hugging Face checkpoint `OpenMOSS-Team/MOSS-TTS-Nano`; audio tokenizer `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`; RTX 5060 Ti; PyTorch `2.12.0+cu130`; BF16; greedy decoding; fixed seed `20260716`; 96 generated audio frames; English reference `assets/audio/en_6.wav`; robust normalization enabled and WeTextProcessing disabled.

**Acceptance criteria:** Inference completes on CUDA; output WAV exists; sample rate is at least 16 kHz; duration exceeds 0.5 s; waveform is finite and non-silent; generated token frames are present; compact evidence is atomically written.

**Commands/artifacts:** `validation/moss_smoke.py`; `validation/run_moss_smoke.sh`; `validation/supervisor/moss-smoke.conf`; remote raw evidence `/workspace/nano-flash-artifacts/g1/moss-smoke/moss-smoke.{json,wav}`.

**Results:** The first attempt exited before model loading because the upstream dynamic model imports `torchaudio`, while the Blackwell image has PyTorch 2.12 and no matching torchaudio 2.12 binary wheel. The available CUDA 13.0 torchaudio index stopped at 2.11, so installing it was rejected as an ABI risk. A narrow, source-controlled SoundFile/SciPy compatibility implementation for `load`, `save`, and `functional.resample` passed its sine-wave self-test. The corrected supervised run produced 96 audio-token frames, 7.680 s of finite non-silent 48 kHz stereo audio, RMS `0.02271471`, peak amplitude `0.29165649`, WAV SHA-256 `c2823803…a5e6b`, 20.3018 s cold elapsed time, and 0.4723 GiB peak allocated VRAM. Every automated smoke check passed. The raw artifacts remain on the instance; the compact metrics and hashes are recorded here because the local sandbox denied a subsequent SCP operation.

**Decision:** PASS. MOSS is viable as the pretrained AR baseline on this GPU. Cold elapsed time is not an inference throughput result and must not be used as RTF; warm stage-level profiling follows in EXP-011.

**Follow-up:** Pin the exact upstream/model revisions in the profiling harness, run warm repeated inference in one loaded process, and measure semantic generation, acoustic decode, RTF, TTFA proxy, VRAM, and determinism.

### EXP-011 — MOSS latency and resource profile

- **Gate:** G1
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16
- **Harness commit:** `2aa8307`

**Goal:** Measure the real warm startup latency, sustained throughput, buffering behavior, stage costs, determinism, and VRAM of the pretrained MOSS AR baseline on the target 16 GB GPU without mixing model-load time into inference latency.

**Configuration:** Upstream commit `11619374849c649486584e3b10ed55b176a924ee`; fully offline local snapshots `44502f…685ebec` for the 117,311,232-parameter TTS model and nested text tokenizer, and `6aa02b…8dd68` for the 21,969,664-parameter audio tokenizer; BF16 TTS weights; SDPA attention; greedy decoding; audio repetition penalty `1.2`; KV cache enabled; seed `20260716`; concurrency 1; English voice clone from `en_6.wav`; one 32-frame warm-up, three uninstrumented measured requests, and one synchronization-instrumented stage request in the same process. The measured cap was 160 frames, but each request ended naturally at 65 frames.

**Acceptance criteria:** At least three valid warm requests plus one valid instrumented request; finite non-silent audio and final result events; identical greedy semantic/audio-token hashes across repeats; less than 14.5 GiB peak allocated VRAM; positive TTFA and RTF measurements; stage timers account for the instrumented request without double-counting. Nano Flash's draft TTFA/RTF targets are comparisons, not this profiling experiment's pass condition.

**Commands/artifacts:** `validation/moss_profile.py`; `validation/run_moss_profile.sh`; `validation/supervisor/moss-profile.conf`; remote raw evidence `/workspace/nano-flash-artifacts/g1/moss-profile/`; summary `/workspace/nano-flash-artifacts/g1/moss-profile/moss-profile.json` with SHA-256 `e3bb7ff8…27288a6`.

**Results:** All automated checks passed. The three warm requests produced identical 65-frame token tensors (SHA-256 `57df80d2…5f24fa`) and byte-identical 5.200 s, 48 kHz stereo WAV files (SHA-256 `35534e23…1e2f9c`); none hit the frame cap. Warm TTFA was `0.1371 s` p50 and `0.1861 s` p95, meeting the draft `<0.150 s` p50 and `<0.250 s` p95 sanity targets in this three-run sample. Sustained RTF was `1.1256` p50 and `1.3186` p95, approximately 5.6 times slower than the `<0.20` target and slower than realtime. Individual maximum inter-audio-event gaps were `92–128 ms` for `80 ms` audio frames, and reported buffer lead became negative on every run (`-0.40` to `-1.54 s`), proving playback underruns rather than merely predicting them from aggregate RTF. The instrumented run attributed `4.1194 s` of `5.7134 s` measured stage time to semantic AR generation (72.1%), `1.5606 s` to codec decode (27.3%), and `0.0334 s` to reference encoding (0.6%); orchestration and WAV I/O added about `0.0625 s`. Peak allocated VRAM was only `0.4723 GiB`, leaving ample memory for larger batches, adapters, or an auxiliary continuation head. Model and component loading took `1.3343 s` and `0.2622 s`, respectively, and were excluded from warm metrics. The three-sample p95 is sanity evidence, not a production percentile claim.

**Decision:** PASS as a reproducible G1 profile. MOSS already demonstrates a viable low-latency AR startup path on the RTX 5060 Ti, so the architecture does not need a new startup model merely to reach first audio. It is not viable as the steady-state path on this GPU: semantic frame generation dominates and the existing codec is secondary. This directly supports the proposed hybrid split—retain AR for startup/fallback and target semantic continuation throughput with block generation—while showing that renderer-only optimization cannot close the full RTF gap.

**Follow-up:** Profile the official Chatterbox baseline under the same timing contract, then use both baselines to set G2/G5 simulator arrival rates and the minimum block-continuation speedup required for realtime operation.

### EXP-012 — Chatterbox access and isolated environment validation

- **Gate:** G1
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16
- **Harness commit:** `b470907`
- **Nano environment harness commit:** `a5d9c13`

**Goal:** Establish reproducible, non-conflicting runtime foundations for the official Chatterbox-Nano and Chatterbox-Flash checkpoints on Blackwell before measuring either model, while proving gated Nano access without storing credentials.

**Configuration:** Official Chatterbox source commit `65b18437192794391a0308a8f705b1e33e633948`; official Chatterbox-Flash source commit `74e05baa8ce574bf2cc571702391a21f1b0d48c5`; official Nano demo Space commit `647b4e895d3483995e5a6546999aa5e50490b92b`; public Flash checkpoint revision `4385507288b8197e6dab8b4e6b1603328d549d9d`; gated Nano checkpoint revision `493317046f21b7e557146a9285a111c050564bb4`. Flash uses a dedicated Python 3.12 environment with PyTorch/torchaudio `2.7.1+cu128`, Transformers `5.2.0`, and pure Torch SDPA. User-authorized Hugging Face credentials were passed only through one process environment and were not written to Git, the ledger, Supervisor, or logs.

**Acceptance criteria:** Exact source revisions pinned; gated Nano repository authorization succeeds and the complete exact snapshot is cached; Flash dependencies remain isolated from MOSS; PyTorch and torchaudio ABI versions match; CUDA 12.8 sees Blackwell compute capability 12.0; a BF16 CUDA matrix multiplication is finite; the official Flash package imports; compact evidence is atomically written.

**Commands/artifacts:** `validation/setup_chatterbox_flash.sh`; `validation/setup_chatterbox_nano.sh`; corresponding Supervisor configurations; remote Flash summary `/workspace/nano-flash-artifacts/g1/chatterbox-flash-setup/environment.json` with SHA-256 `5c22c973…946efc`; remote Nano summary `/workspace/nano-flash-artifacts/g1/chatterbox-nano-setup/environment.json` with SHA-256 `9ec0d807…390556`; gated Nano cache `/workspace/.hf_home/hub/models--ResembleAI--chatterbox-nano/snapshots/493317046f21b7e557146a9285a111c050564bb4`.

**Results:** All environment checks passed. The isolated Flash stack reports RTX 5060 Ti compute capability 12.0, CUDA 12.8, matching PyTorch/torchaudio `2.7.1+cu128`, successful BF16 CUDA execution in `0.2089 s`, and a successful `chatterbox-flash==0.1.0` import. The Flash environment occupies 7.3 GiB. Nano initially returned HTTP 401 as expected for an unaccepted gated model; after the user supplied an authorized token, all 13 files at the pinned 2.8 GiB revision downloaded successfully without credential persistence. Source inspection found an upstream packaging split: the public `chatterbox-tts==0.1.7` GitHub tree does not contain `ChatterboxNanoTTS`; the official Nano loader and Nano-specific T3 inference changes are instead present in the pinned demo Space. The dedicated 5.2 GiB Nano environment subsequently passed every check with matching PyTorch/torchaudio `2.11.0+cu130`, Transformers `4.46.3`, native compute capability 12.0 visibility, a finite BF16 CUDA probe in `0.1733 s`, an offline import of the exact Nano loader, and the pinned snapshot present. Disk remains safe at 285 GiB free. The two isolated stacks therefore reproduce their respective official framework families without contaminating MOSS.

**Decision:** PASS. Do not force Nano and Flash into one environment and do not modify the already validated MOSS environment. Use the official pinned Nano Space source in a dedicated PyTorch 2.11 CUDA 13.0 environment, and use the official Flash repository in its existing PyTorch 2.7 CUDA 12.8 environment. Begin Flash with Torch SDPA; treat FlashInfer as a separate later optimization because it adds compiled ABI and architecture risk.

**Follow-up:** Build and validate the dedicated Nano environment, then run deterministic Nano and Flash smoke/profile experiments using local pinned snapshots and the same structural/latency evidence contract.

### EXP-013 — Chatterbox-Nano smoke and warm profile

- **Gate:** G1
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16
- **Harness commit:** `0b7d4ea`

**Goal:** Validate the official gated Nano checkpoint end to end and measure fixed-seed warm completion/first-audio time, RTF, semantic versus one-step acoustic cost, conditioning cost, determinism, and VRAM on the target GPU.

**Configuration:** Official Nano demo Space commit `647b4e895d3483995e5a6546999aa5e50490b92b`; local gated snapshot revision `493317046f21b7e557146a9285a111c050564bb4`; PyTorch/torchaudio `2.11.0+cu130`; Transformers `4.46.3`; official FP32 loader; cached zero-shot conditioning from the 7.8 s English `en_6.wav` reference; fixed seed `20260716`; canonical Nano sampling parameters; concurrency 1.

**Acceptance criteria:** Model and conditionals load offline; warm-up, at least three uninstrumented requests, and one instrumented request produce finite non-silent audio; fixed-seed semantic tokens repeat; peak allocated VRAM stays below 14.5 GiB; semantic, acoustic, watermark, completion, and RTF measurements are present. The released full-utterance API must be reported honestly as non-streaming.

**Attempt 1:** The model loaded, generated semantic tokens, completed two-step meanflow acoustic inference, and returned audio for the warm-up. The run then failed only when `torchaudio.save` raised `ImportError: TorchCodec is required for save_with_torchcodec`; TorchCodec is not declared by the official Nano requirements. No profile claim was made and no partial JSON was accepted.

**Correction:** Commit `95f4a0c` changed only the evidence writer to the already installed SoundFile library, avoiding an unrelated runtime dependency. The entire experiment was rerun from model load; no measurements from attempt 1 were reused.

**Results:** PASS. The corrected run produced deterministic, byte-identical PCM16 WAV evidence on all three measured requests. Each request generated 114 identical semantic tokens and 4.68 s of finite, non-silent audio. Warm full-completion latency was `1.2405 s` p50 and `1.2523 s` p95; warm RTF was `0.2651` p50 and `0.2676` p95, equivalent to approximately 3.77x real time. The three-run latency coefficient of variation was `0.0090`. Cached conditioning took `1.1995 s`; peak allocated VRAM was `2.7891 GiB`. The instrumented request took `1.2116 s`: semantic generation `1.0313 s` (85.1%), one-step acoustic rendering `0.1473 s` (12.2%), watermarking `0.0329 s` (2.7%), and other overhead `0.0012 s`. Model structure contained 178,859,171 T3 parameters, 266,030,919 S3Gen parameters, and 1,423,618 voice-encoder parameters. The T3 core therefore fits the proposal's 150–220M working range, although the complete deployed pipeline is approximately 446.3M parameters. The raw summary is stored at `/workspace/nano-flash-artifacts/g1/chatterbox-nano-profile/chatterbox-nano-profile.json` with SHA-256 `9d1f0e943c781f916ba59b82214e3530333aa8a383faa78f1b97ef52f2a238a1`.

The released API exposes only full-utterance completion, so its measured first-audio time equals full completion and misses the draft `<250 ms` interactive TTFA target. Its p50 RTF misses the aggressive `<0.20` target by approximately 32.5%, but is comfortably faster than real time and approximately 4.25x faster than the measured MOSS RTF. Percentiles here are sanity statistics over three warm runs, not production percentile estimates.

**Decision:** Nano validates the central reuse hypothesis: its pretrained one-step renderer is fast, deterministic, memory-light, and compatible with the 16 GiB target. It does not validate streaming delivery as released. Semantic generation is the dominant remaining bottleneck, so further work should focus on incremental/block semantic generation and packetized output rather than retraining the renderer from scratch.

**Follow-up:** Profile Chatterbox Flash under the same evidence contract, including its block-generation/streaming surface, then compare MOSS, Nano, and Flash before selecting components for the lean hybrid path.

### EXP-014 — Chatterbox-Flash smoke and warm profile

- **Gate:** G1
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Validate the official public Chatterbox-Flash checkpoint end to end on the RTX 5060 Ti and measure deterministic warm full-completion latency, RTF, block-diffusion semantic cost, two-step acoustic cost, conditioning, and VRAM under the pure Torch SDPA backend.

**Configuration:** Official Flash source commit `74e05baa8ce574bf2cc571702391a21f1b0d48c5`; public checkpoint revision `4385507288b8197e6dab8b4e6b1603328d549d9d`; isolated PyTorch/torchaudio `2.7.1+cu128` environment; BF16 T3; DRF block size 16; 10 diffusion steps; Torch backend without CUDA graphs; two meanflow acoustic steps; cached English zero-shot conditioning; fixed seed `20260716`; concurrency 1.

**Acceptance criteria:** The pinned snapshot downloads and loads offline; warm-up, at least three measured requests, and one instrumented request produce finite non-silent audio; fixed-seed semantic tokens repeat; peak allocated VRAM stays below 14.5 GiB; semantic and acoustic stage timings are present. Source inspection found no waveform-yielding or `generate_stream` method, so full-completion time will be reported as first-audio time and native streaming will not be claimed.

**Results:** PASS as a pure-Torch architectural baseline. The exact 3,200,254,266-byte public snapshot downloaded and resolved to the requested revision. All warm and instrumented requests produced finite non-silent 24 kHz audio, 126 identical semantic tokens, and byte-identical 5.04 s PCM16 WAVs. Warm full-completion latency was `2.9825 s` p50 and `3.0025 s` p95; warm RTF was `0.5918` p50 and `0.5957` p95. The three-run latency coefficient of variation was `0.0044`. Peak allocated VRAM was only `2.4128 GiB`. The instrumented request took `2.9816 s`: block-diffusion semantic generation `2.8298 s` (94.9%), two-step meanflow acoustic rendering `0.1507 s` (5.1%), and orchestration `0.0011 s`. Initial conditioning took `14.8466 s` and peaked at `2.2594 GiB`; this is an offline per-voice preparation cost and was excluded from warm request latency. The model contains 532,406,272 T3 parameters, 266,030,919 S3Gen parameters, and 1,423,618 voice-encoder parameters.

The raw summary is stored at `/workspace/nano-flash-artifacts/g1/chatterbox-flash-profile/chatterbox-flash-profile.json` with SHA-256 `705813c2d251c32d5e5c62a48409b48339fb4fff9df965fff83b22801b192f84`. Evidence occupies approximately 1 MiB, and 282 GiB remains free. As with EXP-013, p50/p95 values are three-run sanity statistics rather than production percentile estimates.

**Decision:** The released Flash components work on the 16 GiB GPU, but the pure Torch path misses both latency targets and is slower than Nano. This result does not invalidate block diffusion: the repository's optimized claims depend on FlashInfer and CUDA graphs, neither of which was enabled in this compatibility-first baseline. It does show that the semantic model—not the renderer—is again the dominant cost, and that the public wrapper's complete-waveform return prevents native interactive TTFA measurement. Reuse the pretrained Flash/Nano components; do not train either stack from scratch. Treat FlashInfer/CUDA graphs and true block-to-audio scheduling as explicit optimization/streaming gates rather than assumed capabilities.

**Follow-up:** Close G1 with a comparative component decision, then implement the G2 incremental-text simulator. Later G7 optimization will test whether FlashInfer/CUDA graphs materially change the block baseline on Blackwell before selecting or adapting a frozen block head.

### EXP-020 — Incremental-text simulator and prompt suite

- **Gate:** G2
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Prove that English text arriving in arbitrary fragments can be converted into an append-only committed prefix without revising text associated with audio already released to the listener.

**Configuration:** Ten English prompt classes covering plain prose, currency/date numbers, abbreviations, email, URL, contractions, quotation, Unicode punctuation, time notation, and long-form input. Each prompt was split 25 ways using deterministic random chunk widths of 1–9 characters. The committer retains two completed words of lookahead but may commit earlier at clause or sentence boundaries. Only prefix-stable surface normalization is allowed online; context-sensitive verbalization remains deferred.

**Acceptance criteria:** Every intermediate committed value extends the previous value; committed plus pending text exactly reconstructs all normalized received text; finalization is lossless; incomplete tails retain two words; punctuation and Unicode surface normalization behave deterministically.

**Results:** PASS. All four test groups passed. Across 250 randomized prompt/chunking combinations, every committed prefix was monotonic and every finalized string exactly matched normalized input. Dedicated tests confirmed a two-word incomplete tail, immediate sentence-boundary commitment, and stable conversion of curly quotes and em dashes. An initial unit-test expectation incorrectly assumed three held words; the implementation correctly held the configured two words, and only the assertion was corrected before accepting the run.

**Decision:** Incremental English commitment is mechanically feasible without model training, provided the online layer limits itself to prefix-stable normalization and maintains lookahead. Context-sensitive expansions such as abbreviations, currency, dates, and ambiguous numbers must be performed only inside the uncommitted region or by a downstream tokenizer with an explicit stability contract.

**Follow-up:** EXP-021 will replay the suite at multiple text-arrival rates against measured component service times, quantifying commitment delay, buffer occupancy, underruns, and the AR/block switching region.

### EXP-021 — Partial-text stability across arrival rates

- **Gate:** G2
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Quantify the latency cost of stable two-word lookahead when English text arrives incrementally, and distinguish upstream text starvation from GPU/model latency.

**Configuration:** The ten-prompt EXP-020 suite was normalized first and replayed at 5, 10, 20, and 40 characters per second on a 20 ms clock. Every newly committed character was timestamped. Forty scenarios measured first commitment, p50/p95 character commitment lag, event count, and maximum pending characters.

**Acceptance criteria:** All 40 scenarios finalize losslessly, every character receives a nonnegative commit timestamp, every arrival rate is represented, and pending text remains bounded.

**Results:** PASS for stability, with a product-significant latency finding. All 40 scenarios were lossless and bounded; maximum pending text was 23 characters at every rate. Median first-commit time was `2.10 s` at 5 chars/s, `1.05 s` at 10 chars/s, `0.53 s` at 20 chars/s, and `0.27 s` at 40 chars/s. Across prompts, median character commitment lag was respectively `1.40`, `0.70`, `0.35`, and `0.18 s`; aggregate p95 commitment lag was `4.2825`, `2.1413`, `1.0765`, and `0.5283 s`. Raw local evidence SHA-256 was `da4dd1e196d6014b37894ad534361d28387d71c9612b741f995ec88e758b09e1`.

**Decision:** Stable committed text alone cannot deliver the `<150 ms` TTFA target, even at a fast 40 chars/s input stream, because two-word lookahead delays the first safe prefix to 270 ms before inference begins. The architecture therefore needs two confidence classes: speculative/revocable text may feed the AR startup path before full commitment, while block generation must consume only stable committed spans. Playback release needs a short revision window or explicit LLM token-commit signal. At slow arrival rates the input source, not the GPU, is necessarily the limiting factor; underrun metrics must be conditioned on supplied-text rate.

**Follow-up:** Preserve the stable committer as the block-path contract. In G3/G4, validate semantic token and training pipelines; in G5, simulate speculative AR startup, stable block continuation, buffer thresholds, cancellation, and rollback before releasing packets.

### EXP-030 — Token/data pipeline integrity

- **Gate:** G3
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Validate the official pretrained MOSS audio-tokenizer and teacher-forcing data path before spending GPU time on optimization, using a deliberately repeated English corpus that removes data diversity as a confounder.

**Configuration:** One clean 7.8 s English utterance and transcript from the pinned MOSS repository was expanded into 100 uniquely identified records. The exact pinned MOSS audio tokenizer encoded targets with one CUDA process. The official SFT dataset packed records at maximum length 512 using the separately pinned custom MOSS text tokenizer and model configuration.

**Acceptance criteria:** Exactly 100 unique English records; deterministic rank-two nonnegative audio codes with the model's 16-codebook width; loss-bearing teacher-forcing labels; prompt/padding masking; valid fixed-size collation; identical input records pack identically.

**Attempt 1:** Audio encoding completed successfully, but the validator tried to load the custom tokenizer from the model snapshot. MOSS keeps its tokenizer implementation in a separate cached immutable snapshot, so packing stopped before making a claim. Commit `cf4d486` supplied the already validated G1 tokenizer snapshot; the complete pipeline was rerun.

**Results:** PASS. Every record encoded to a deterministic `97 x 16` audio-token tensor with values from 0 to 1022 and SHA-256 `4e9d207b8fdf3666df80236d75cec58c0941c0e604acfabe00e12bb0f4644a39`. Each official packed sequence had length 243 with a 145-step prompt. Two-sample collation produced shape `2 x 511 x 17`, contained 3,300 active supervised values, correctly masked prompt/padding targets, and packed the first and last repeated records identically. All eleven integrity checks passed. Integrity JSON SHA-256 is `9894678a4d78014acb35a3cf4770f24ebf1239040c89e0de9f3894434d2556c5`; prepared JSONL SHA-256 is `3b25b84f937496d9a199757bcc8c227457b7ae3474b252dc8be87a964b6eb084`. Total evidence occupies approximately 840 KiB.

**Decision:** The official pretrained tokenizer and SFT packing/masking pipeline are suitable for the lean training gates. Repeating one utterance is intentional only for plumbing and overfit validation; it provides no evidence of generalization or production voice quality.

**Follow-up:** Run EXP-031 as a short full-model deliberate overfit with loss-decrease and finite-gradient criteria, retaining only compact checkpoints. Then prove restart equivalence in EXP-032.

### EXP-031 — 100-sample deliberate overfit

- **Gate:** G3
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Prove that the complete pretrained MOSS semantic model can backpropagate through the official packed English data, rapidly memorize a deliberately trivial corpus, and emit a finite reloadable checkpoint on the 16 GiB GPU.

**Configuration:** Full-model BF16 SFT; 100 repeated English records from EXP-030; maximum length 512; batch size 1; 40 optimizer steps; AdamW learning rate `1e-4`; no weight decay or warmup; linear decay; gradient norm cap 1; SDPA; audio-weight total 32 versus text weight 1; seed `20260716`.

**Attempt 1:** Training stopped before the first optimizer step because the official trainer assumes model and tokenizer assets share one directory, while the published checkpoint separates them. Commit `f2f33f9` created a deterministic merged local model view from the two pinned snapshots without altering upstream source or weights, and the same training configuration was rerun.

**Results:** PASS. All 40 steps logged finite loss. Loss decreased monotonically from `5.6598` at step 1 to `0.5437` at step 40, a final/initial ratio of `0.0961` or 90.4% reduction. The first measured step took 0.95 s, while the warmed final step took 0.10 s. The reloadable checkpoint contains 194 finite tensors, occupies 285,015,275 bytes, and has SHA-256 `8a363d330a35b9a0a2aaafecaa0b682b5bbd26fb9906282f6bf26f8723918cf8`. All ten acceptance checks passed. Summary JSON SHA-256 is `71c16cf73c9132eb2f90055523a078eeacdd9cf7596dd8276c0be4ad480d1589`.

**Decision:** The pretrained MOSS training stack is learnable and stable on the RTX 5060 Ti with substantial memory headroom, so training from scratch is unnecessary for the lean validation. This experiment demonstrates optimization/plumbing only; because every record is the same utterance, it provides zero generalization or production-quality evidence.

**Follow-up:** EXP-032 will test checkpoint reload and continuation. The upstream trainer's checkpoint format currently saves model/config/tokenizer but not optimizer, scheduler, or RNG state; exact interrupted-run equivalence must therefore be treated as unproven unless the harness adds stateful recovery.

### EXP-032 — Checkpoint resume and reproducibility

- **Gate:** G3
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Demonstrate that unattended single-GPU training can resume after interruption without changing the next optimization result, including model, optimizer, scheduler, and random-number state.

**Configuration:** Start from the EXP-031 overfit checkpoint; one deterministic English batch; BF16 model; AdamW at `1e-5`; constant scheduler; SDPA math backend; seed `20260716`. Execute step 1, atomically save full recovery state, execute step 2 uninterrupted, destroy the live objects, reload fresh model/optimizer/scheduler objects and RNG state, then execute the same step 2 and hash all resulting tensor state.

**Acceptance criteria:** All losses finite; uninterrupted and resumed step-2 losses exactly equal; post-step model and optimizer state hashes exactly equal; state file atomically published below 2 GiB; peak VRAM below 14.5 GiB.

**Attempts 1–2:** Both training paths executed, but evidence hashing first failed on NumPy's lack of BF16 support and then on byte-viewing scalar optimizer tensors. Commits `cee84c3` and `0cc70b5` changed only raw-byte hashing. No partial recovery result was accepted; the complete comparison was rerun after each correction.

**Results:** PASS. Step-1 loss was `0.5441401601`. Uninterrupted and restored step-2 loss were exactly `0.5428910255`. Both paths produced model SHA-256 `43b87cabf645bcbb6f71e267092b8a3264d09686e1460d573356f6801765190e` and optimizer SHA-256 `11de9af96351c6312c197a95a90f26c0a8e620667615056c63c9ae8a0a26f6d2`. The atomic full-state checkpoint is 754,421,427 bytes with SHA-256 `7c1da1b803786ff7729757cacbbe6accdb2623a91b6a62e89491880394d626ef`. Peak allocated VRAM was 1.7264 GiB uninterrupted and 1.7277 GiB restored. Summary JSON SHA-256 is `46da1c680c103ecc2e5ed6b94f97988a76ae712521ef2db10eb35ec4b29ff0f2`. Disk remains safe with 280 GiB free.

**Decision:** Exact recovery is feasible, but only when the harness saves optimizer, scheduler, and RNG state in addition to upstream model-only checkpoints. All longer unattended training gates must use this full-state atomic contract. Keep only last, best, and one recovery state to control disk use.

**Follow-up:** G3 is complete. Begin G4 lean English adaptation/generalization with a small real multi-utterance corpus and held-out evaluation; do not interpret the repeated-sample overfit checkpoint as a candidate model.

### EXP-040 — Lean English AR adaptation

- **Gate:** G4
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Determine whether a cost-effective, short English adaptation of the pretrained AR semantic model trains stably on real varied speech and improves rather than degrades loss on unseen speakers.

**Configuration:** Deterministic LibriSpeech `clean` subset selected through the Hugging Face Dataset Viewer API: 100 `train.100` utterances distributed across offsets 0/5000/10000/15000/20000 and 20 validation utterances across offsets 0/500/1000/1500. Training contains 21.45 minutes from five speakers; held-out contains 2.44 minutes from four disjoint speakers. Dataset license is CC BY 4.0. Full pretrained MOSS model; BF16; 100 steps; batch 1; maximum length 512; AdamW `1e-5`, weight decay 0.01, five warmup steps, linear decay, gradient cap 1, seed `20260716`.

**Acceptance criteria:** Exactly 100/20 finite 16 kHz English clips with unique IDs and disjoint speakers; deterministic 16-codebook encoding; 100 finite optimizer steps; late training loss below early training loss; finite held-out evaluation before and after; adapted held-out mean loss below baseline; reloadable checkpoint.

**Results:** PASS, with a small effect size. All 120 audio files passed structural checks. The train and validation token manifests contain respectively 100 and 20 records, with frame ranges 21–208 and 24–238. Early ten-step training loss averaged `5.10049`; late ten-step loss averaged `5.06226`. Mean loss on four held-out speakers improved from `5.0763384` to `5.0548172`, a ratio of `0.9957605` or 0.424% improvement. The 285,015,275-byte checkpoint has SHA-256 `9f9838e06d7845b63d17ea65250cbeca50e75b1586d2e03465dd8fd3853e12bb`. Provenance, encoding, and adaptation summary SHA-256 values are respectively `a79309635ffd8febc0e863d2b58f39488f3b1e0f1e1c68f7833f3a01b3d20e08`, `ef1eb6cdbb3c72bf2b50029a171f05aa00d75cbda746c315c12189264a26d681`, and `519646e6bf98f3e958048d3e0bd91706b6e362d6d36afd4c510d98fc555e527e`. G4 artifacts occupy approximately 572 MiB; 280 GiB remains free.

**Decision:** Real English adaptation is technically feasible and stable on the target GPU, but 100 steps/21 minutes yields only marginal held-out improvement and should not be presented as a meaningful quality gain. The result supports reuse and low-cost adaptation rather than training from scratch. Candidate selection must depend on generated-speech intelligibility and failure analysis, not teacher-forced loss alone.

**Follow-up:** EXP-041 will synthesize a held-out English challenge set with baseline and adapted checkpoints, transcribe outputs using a frozen ASR evaluator, compare WER/CER and structural failures, and retain samples for manual listening.

### EXP-041 — Held-out intelligibility and failure analysis

- **Gate:** G4
- **Status:** FAIL
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Determine whether the marginal teacher-forced gain from EXP-040 translates into non-degraded generated English speech on unseen challenge text.

**Configuration:** Ten deterministic English prompts spanning plain prose, numbers, dates, names, initialisms, contrast, contractions, punctuation, rare words, and a long sentence. Both the untouched pretrained model and EXP-040 checkpoint used the same English voice reference, greedy decoding, seed, tokenizer, codec, and 192-frame ceiling. All twenty WAVs were transcribed by frozen `openai/whisper-tiny.en` revision `87c7102498dcde7456f24cfd30239ca606ed9063`. Scoring lowercased and removed punctuation but did not semantically normalize numeric renderings; paired comparison therefore remains valid even where absolute WER is conservative.

**Acceptance criteria:** Ten finite, non-silent, structurally valid outputs per checkpoint; twenty ASR transcripts; WER and CER below 100%; adapted WER/CER no more than five absolute points worse than baseline.

**Attempts 1–2:** Synthesis passed immediately. ASR scoring first found that Librosa was absent from the isolated MOSS environment, then found an API incompatibility in `Tensor.cuda(dtype=...)`. Commits `9ae2e60` and `f50702e` replaced Librosa with validated SoundFile/SciPy resampling and used `.to(device, dtype)`. Synthesis was cached and not regenerated; the complete frozen-ASR comparison was rerun.

**Results:** FAIL. All twenty WAVs passed structural checks. Baseline WER was `0.19` and CER `0.18994`; adapted WER was `0.32` and CER `0.33333`, regressions of 13.0 and 14.34 absolute points. Both models were perfect on plain, initialism, contrast, and rare-word cases. Number/date errors were identical and partly reflect ASR numeric formatting. The adapted checkpoint additionally changed “Maya” to “Meyer” and, critically, truncated the punctuation prompt after “Wait” and the long prompt after its first clause. Baseline completed the long prompt with zero normalized ASR errors. Comparison JSON SHA-256 is `726825f459482d3250c5ddbe13ae56c65da8fe61c172d66c158c35e84b3f3cf2`; baseline/adapted synthesis manifests are `4d6cbf4258b7198b508d7c30f7edcb0d9c0374d2be28e5da1a036d8266c9e495` and `565ed5c993bf80f4d10b529c8c0fd12e811ffb08a7a9b72c40a6fb187504df1c`. All WAVs are retained for manual listening.

**Decision:** Reject the EXP-040 adapted checkpoint. A 0.424% teacher-forced loss improvement did not predict generation quality and concealed severe early-EOS/truncation regression. Retain the untouched pretrained MOSS model as the AR startup baseline. Any later adaptation must include EOS-aware sampling, longer/more varied data, and generation-based checkpoint selection; training from scratch remains unjustified.

**Follow-up:** G4 is complete with a negative candidate-selection result. Build G5 around the retained pretrained AR measurements, Nano renderer timings, stable/unstable text classes, packet buffering, cancellation, and explicit underrun accounting.

### EXP-050 — Streaming scheduler and packetizer

- **Gate:** G5
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Establish deterministic runtime invariants before combining model timing: lossless 80 ms PCM packetization, bounded playback-buffer accounting, speculative-text isolation, stable block eligibility, low-buffer AR fallback, and immediate cancellation.

**Configuration:** 24 kHz mono float PCM; 1,920-sample/80 ms packets; 640 ms nominal semantic blocks; 240 ms block-switch buffer threshold. AR may consume speculative text; block mode may consume only stable text. Cancellation discards all queued future audio and permanently disables further scheduling for that request.

**Acceptance criteria:** Fragmentation-independent lossless packetization including final tail; block selection only with sufficient buffer and stable span; AR fallback below threshold; no negative buffer; cancellation empties queued audio and rejects subsequent generation.

**Results:** PASS. Seven unit tests completed in 31 ms. Packetizing 5,000 samples produced exact lengths 1,920/1,920/1,160 and reconstructed the source byte-for-byte. Fragmenting a 7,680-sample input into 137-sample writes produced exactly the same packets as a single write. All routing, buffer, and cancellation invariants passed.

**Decision:** The serving-state contract is implementable independently of model internals. Speculative text is confined to AR startup; stable text and adequate playback headroom are mandatory for block work. This protects already released audio from text revision and makes cancellation behavior explicit.

**Follow-up:** EXP-051 will drive this state machine with measured MOSS AR and Flash block timing over arrival rates, block sizes, switch thresholds, cancellations, and component-speed counterfactuals to find the feasible operating region.

### EXP-051 — TTFA/RTF/gap/cancellation stress matrix

- **Gate:** G5
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Determine whether the measured components can simultaneously satisfy startup, continuous playback, text-arrival, switching, and cancellation constraints, and quantify the component-speed region required for feasibility.

**Configuration:** Discrete 5 ms clock; 6 s target utterance; 15 arriving characters per second of spoken audio; seven-character stability lookahead; arrival rates 5/10/20/40/80/1000 chars/s; block sizes 160/320/640 ms; switch buffers 80/160/240 ms; cancellation at none/1.0/2.5 s. Measured profile uses MOSS first packet 137.1 ms, AR RTF 1.1256, and Flash Torch block RTF 0.5918. Required-optimized counterfactual uses 100 ms first packet, AR RTF 0.75, and block RTF 0.20. The full Cartesian matrix contains 324 scenarios.

**Acceptance criteria:** Complete matrix; nonnegative buffers; all cancellations within one 80 ms packet; no measured scenario falsely reported feasible under adequate text; at least one optimized feasible region; slow-input starvation identified separately from model underrun.

**Results:** PASS as an architectural stress test, with the measured stack failing the product envelope. All six harness checks passed. Measured components produced zero product-passing scenarios, zero block transitions, and a best underrun ratio of `0.1561` even among 36 text-sufficient cases; minimum TTFA was 145 ms only with essentially complete text availability. The slower-than-real-time AR path cannot accumulate the 80–240 ms buffer needed to hand off, creating a switching deadlock. The required-optimized profile passed all 36 text-sufficient cases and 45/54 uncancelled scenarios overall, reached block mode in 27 scenarios, achieved zero underrun in its best cases, and minimum TTFA 105 ms. Every cancellation completed within 80 ms. Rates below the 15 chars/s content-consumption rate were correctly classified as input-starved rather than model failures. Raw matrix SHA-256 is `00d4b2f72b4108c333e377dd7ebc0e97bb0ee3af02c057c50094c122236dc0a8` and occupies approximately 256 KiB.

**Decision:** The hybrid scheduler is viable, but the currently measured MOSS/Flash combination is not. To avoid handoff deadlock, AR continuation must be at least modestly faster than real time (validated counterfactual RTF 0.75) while block generation approaches RTF 0.20. Alternatively, block work must run concurrently or start from a much smaller incremental unit, neither of which the released APIs currently expose. TTFA cannot be guaranteed when upstream text arrives below speech consumption rate.

**Follow-up:** G5 is complete. In G6, validate that pretrained Nano acoustic targets can be cached and that its one-step renderer is already sufficient. In G7, test FlashInfer/CUDA-graph acceleration and block-size behavior before deciding whether a frozen block head needs lean training.

### EXP-060 — Renderer target caching

- **Gate:** G6
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Prove that expensive semantic generation can be removed from renderer experiments by caching all inputs needed to reproduce Nano's raw one-step acoustic output exactly.

**Configuration:** Official Nano snapshot and English reference conditioning; ten EXP-041 challenge prompts; fixed seed; semantic tokens captured at the S3Gen boundary; raw pre-watermark waveform captured as target; cached conditionals; two meanflow steps. Each cache item was reloaded and rendered through S3Gen alone.

**Acceptance criteria:** Ten entries with nonempty semantic tokens; conditionals cached; every cached raw waveform exactly equals rerendered output; every renderer-only RTF below 0.20.

**Attempt 1:** Tokens and conditioning rerendered quickly but not byte-identically because S3Gen consumes stochastic noise after semantic sampling has advanced the RNG. This established that tokens alone are an incomplete cache contract. Commit `c46fd27` added the CPU/CUDA RNG state at the renderer boundary; all ten items were regenerated and retested.

**Results:** PASS. All ten cached waveforms rerendered bit-for-bit after restoring semantic tokens, conditioning, and exact renderer RNG state. Mean renderer-only RTF was `0.03822`, approximately 26.2x real time; every item was below 0.20. Total cache size including conditioning was 4,063,243 bytes (approximately 3.9 MiB). Summary JSON SHA-256 is `2671043080c64d62a568091496d8346996e915dace1fbc16bd5fa96341f41381`.

**Decision:** Nano's pretrained renderer is already substantially faster than the end-to-end target and deterministic under a complete cache contract. Do not train or distill a replacement during lean validation. Cache tokens, conditioning, and renderer noise state for all later renderer/block experiments; watermarking remains a separate postprocess.

**Follow-up:** EXP-061 will compare one versus two meanflow steps on the same cached targets, quantifying speed and objective waveform/spectral deviation. Manual listening remains required before any one-step quality claim.

### EXP-061 — One-step/two-step renderer feasibility

- **Gate:** G6
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Test whether reducing Nano meanflow rendering from its released two steps to one produces valid, faster audio and quantify objective deviation from exact cached two-step targets.

**Configuration:** Ten EXP-060 caches with identical semantic tokens, reference dictionary, and renderer RNG state. Render each item with two steps to revalidate the cache and with one step from the identical starting noise. Measure RTF, speedup, normalized RMSE, cosine similarity, log-spectral L1, and SNR.

**Acceptance criteria:** Ten pairs; exact two-step cache reproduction; finite one-step output with identical shape; every one-step RTF below 0.20. Objective similarity is reported rather than used as an automatic quality pass because waveform error is not a perceptual listening metric.

**Attempt 1:** The cached reference dictionary was passed positionally into the `ref_wav` argument, so rendering stopped before measurement. Commit `36a1cab` used explicit `speech_tokens=` and `ref_dict=` keywords; the complete comparison was rerun.

**Results:** PASS for structural/speed feasibility. Two-step targets reproduced exactly. Mean two-step RTF was `0.06565`; one-step RTF was `0.02350`, a `2.5268x` speedup. However, one-step output differed substantially from the released target: mean cosine similarity `0.77165`, normalized RMSE `0.66413`, SNR `3.7258 dB`, and log-spectral L1 `0.01464`. Summary JSON SHA-256 is `8ff6193291163c7e6a4898c5ecae1530c1979b194a7366f4170a1f5c22822882`.

**Decision:** One-step execution works and is very fast, but objective deviation is too large for an automatic quality endorsement. The released two-step renderer already consumes far below the total RTF budget, so retain two steps as the lean default. Consider one step only after blinded listening or a perceptual metric demonstrates acceptable quality; renderer optimization is not the critical bottleneck.

**Follow-up:** G6 is complete. Move to G7 block acceleration: first test whether FlashInfer can be installed and run on Blackwell, then compare Torch/CUDA-graph/block sizes. Only train a frozen block head if optimized pretrained inference remains short of the required RTF.

### EXP-069 — FlashInfer Blackwell compatibility

- **Gate:** G7
- **Status:** PASS
- **Started:** 2026-07-16
- **Finished:** 2026-07-16

**Goal:** Determine whether the official Flash optimized backend can be installed on the RTX 5060 Ti without breaking the already validated Torch/CUDA ABI, and execute a real fused CUDA kernel before trusting benchmark claims.

**Configuration:** Existing isolated Chatterbox-Flash Python 3.12 environment; repository lockfile version `flashinfer-python==0.6.11.post3`; explicitly preserved `torch==2.7.1+cu128`; FlashInfer cache rooted under `/workspace`; BF16 256x512 fused add+RMSNorm probe.

**Acceptance criteria:** Torch and CUDA versions unchanged; compute capability 12.0 visible; exact FlashInfer version installed; repository engine detects backend; fused CUDA output finite.

**Results:** PASS. All six checks passed. FlashInfer 0.6.11.post3 installed while retaining Torch 2.7.1+cu128 and CUDA 12.8. The repository engine reported availability, Blackwell compute capability 12.0 was visible, and a real fused BF16 add+RMSNorm kernel returned finite output. Its cold first invocation, including compilation/loading overhead, took `0.9638 s`. Setup JSON SHA-256 is `0a8f1016ab5ea6518f90c493b8734d47f966d0cb0dfc499b16d906e07b7a317e`. The Flash environment occupies 7.7 GiB and 279 GiB remains free.

**Decision:** FlashInfer is technically viable on this machine and may now be benchmarked. Cold JIT latency must be excluded only after explicit warmup; serving images need precompiled/warmed kernels or cold-start reporting. No block-head training is justified until optimized pretrained inference is measured.

**Follow-up:** Benchmark Torch versus FlashInfer with and without CUDA graphs and across block sizes using identical tokens/audio/seed. Record cold warmup separately and test numerical/output consistency.
