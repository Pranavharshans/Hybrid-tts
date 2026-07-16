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
| EXP-002 | G0 | GPU/CUDA/PyTorch capability validation | PLANNED | — |
| EXP-003 | G0 | Disk, network, checkpoint, and recovery validation | PLANNED | — |
| EXP-010 | G1 | MOSS-TTS-Nano installation and smoke inference | PLANNED | — |
| EXP-011 | G1 | MOSS latency and resource profile | PLANNED | — |
| EXP-012 | G1 | Chatterbox installation and smoke inference | PLANNED | — |
| EXP-013 | G1 | Chatterbox latency and resource profile | PLANNED | — |
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
