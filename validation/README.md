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
