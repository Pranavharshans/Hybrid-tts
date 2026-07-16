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
| EXP-013 | G1 | Chatterbox-Nano smoke and warm profile | PLANNED | — |
| EXP-014 | G1 | Chatterbox-Flash smoke and warm profile | PLANNED | — |
| EXP-020 | G2 | Incremental-text simulator and prompt suite | PLANNED | — |
| EXP-021 | G2 | Partial-text stability across arrival rates | PLANNED | — |
| EXP-030 | G3 | Token/data pipeline integrity | PLANNED | — |
| EXP-031 | G3 | 100-sample deliberate overfit | PLANNED | — |
| EXP-032 | G3 | Checkpoint resume and reproducibility | PLANNED | — |
| EXP-040 | G4 | Lean English AR adaptation | PLANNED | — |
| EXP-041 | G4 | Held-out intelligibility and failure analysis | PLANNED | — |
| EXP-050 | G5 | Streaming scheduler and packetizer | PLANNED | — |
| EXP-051 | G5 | TTFA/RTF/gap/cancellation stress matrix | PLANNED | — |
| EXP-060 | G6 | Renderer target caching | PLANNED | — |
| EXP-061 | G6 | One-step/two-step renderer feasibility | PLANNED | — |
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
