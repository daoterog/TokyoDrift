# Particle-system drift benchmarks

A self-contained implementation of alignment-free, unnormalized kernel drift on the DW4 and LJ13 particle systems from *Equivariant Flow Matching* (Klein, Krämer, and Noé, 2023). It deliberately excludes QM9 and all chemistry-specific code.

The default method uses a single Gaussian kernel on permutation-, rotation-, reflection-, and translation-invariant sorted pair-distance descriptors. The generator is E(n)-equivariant and directly maps centered noise to a particle configuration.

## Layout

```text
particle_systems/
├── configs/          # tracked, grouped by system
├── data/             # local datasets; ignored by Git
├── artifacts/        # checkpoints and evaluations; ignored by Git
├── scripts/          # platform setup helpers
├── tests/            # unit tests
└── *.py              # training, evaluation, model, drift, and system modules
```

`configs/dw4/gaussian.json` and `configs/lj13/gaussian.json` are the baseline configurations.

## Setup and data

```bash
uv sync --project particle_systems
uv run --project particle_systems --no-sync python -m particle_systems.prepare_data dw4
```

The data command downloads the official array, verifies its checksum, centers configurations, and writes deterministic train/test splits under `data/`.

## Train

```bash
uv run --project particle_systems --no-sync python -m particle_systems.train \
  --config particle_systems/configs/dw4/gaussian.json
```

Every run must use its own `artifacts/runs/...` directory. Do not run or resume two processes against the same directory.

## Evaluate

```bash
uv run --project particle_systems --no-sync python -m particle_systems.evaluate \
  --checkpoint particle_systems/artifacts/runs/dw4/gaussian/best_validation.pt \
  --output particle_systems/artifacts/evaluations/dw4-seed42 \
  --num-samples 500000
```

## GMM-40 benchmark

```bash
uv run --project particle_systems --no-sync python -m particle_systems.toys.gmm40 \
  --output particle_systems/artifacts/runs/gmm40/gaussian

# Long-tailed GMM-40: validity and rare-mode retention, not exact mass matching
uv run --project particle_systems --no-sync python -m particle_systems.toys.gmm40 \
  --distribution imbalanced \
  --output particle_systems/artifacts/runs/gmm40/imbalanced-seed42

uv run --project particle_systems --no-sync python -m particle_systems.toys.gmm40_report \
  --run particle_systems/artifacts/runs/gmm40/imbalanced-seed42 \
  --distribution imbalanced

# Validity-first DW4 audit: collision-free and below a declared broad energy cutoff.
uv run --project particle_systems --no-sync python -m particle_systems.evaluate \
  --checkpoint particle_systems/artifacts/runs/dw4/gaussian-batch2048/checkpoint-030000.pt \
  --output particle_systems/artifacts/evaluations/dw4/gaussian-batch2048 \
  --valid-energy-quantile 0.99 --min-pair-distance 0.5
```

This exact 2D target reports per-mode mass error, mode coverage, and sliced Wasserstein-2. It is a fast multimodal complement to the equivariant particle-system benchmarks.

## Test

```bash
uv run --project particle_systems --no-sync python -m unittest discover -s particle_systems/tests -v
```
