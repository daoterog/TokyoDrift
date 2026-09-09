# Particle-system drift benchmarks

A self-contained implementation of direct-coordinate, unnormalized Gaussian-kernel drift on
the DW4, LJ13, and LJ55 particle systems from *Equivariant Flow Matching* (Klein, Krämer, and Noé,
2023). It deliberately excludes QM9 and all chemistry-specific code.

The method applies the analytical gradient of a Gaussian kernel directly to flattened particle
coordinates. It uses data attraction minus generated-sample repulsion without dividing by local
kernel mass. The generator remains E(n)-equivariant and maps centered noise directly to a
particle configuration, but the drift comparison itself depends on particle ordering and global
orientation.

## Layout

```text
particle_systems/
├── configs/          # tracked, grouped by system
├── data/             # local datasets; ignored by Git
├── artifacts/        # checkpoints and evaluations; ignored by Git
├── jobs/             # compute-node environment and dataset download jobs
├── scripts/          # platform setup helpers
├── tests/            # unit tests
└── *.py              # training, evaluation, model, drift, and system modules
```

Each system has a baseline configuration at `configs/<system>/gaussian.json`.

## Setup and data

```bash
uv sync --project particle_systems
uv run --project particle_systems --no-sync python -m particle_systems.prepare_data dw4
```

The data command downloads the official array, verifies its checksum, centers configurations, and writes deterministic train/test splits under `data/`.

For a compute node, install the locked CUDA environment and prepare every dataset with:

```bash
sbatch particle_systems/jobs/download_env.sh
# Submit this after the environment job succeeds.
sbatch particle_systems/jobs/download_datasets.sh
```

Run `bash particle_systems/jobs/download_env.sh cpu` instead when CUDA is not required. LJ55 is
distributed as two 3.3 GB arrays, so downloading all source data requires at least 8.3 GB plus
space for the prepared archives.

## Train

```bash
uv run --project particle_systems --no-sync python -m particle_systems.train \
  --config particle_systems/configs/dw4/gaussian.json
```

Every run must use its own `artifacts/runs/...` directory. Do not run or resume two processes against the same directory.

Training is measured in full epochs. At the start of every epoch, the training references are
reshuffled and divided into batches without replacement; the final partial batch is retained, so
every training configuration contributes exactly once per epoch. `positive_references` controls
the reference-batch size, while `batch_size` controls how many generated configurations are
updated against each reference batch. Logging, validation, learning-rate and bandwidth schedules,
and checkpoint intervals are all expressed in epochs.

`training.bandwidth` accepts a positive number, `"auto"` for the median-distance heuristic, or a
nonempty list of positive numbers. With a list, the drift and kernel-mass diagnostics are the
equal-weight averages of the independently computed Gaussian fields. Bandwidth schedules, when
present, multiply every value in the list by the same epoch-dependent scale.

The large DW4 configuration can be trained and evaluated in one GPU job:

```bash
sbatch particle_systems/jobs/train_dw4.sh
```

The job stores its final checkpoint under `models/dw4/`. Every model run also keeps the fully
resolved training parameters, including command-line overrides and derived bandwidths, in
`models/dw4/runs/<job-id>/parameters.json`. Metrics, plots, and the combined job log are written
under `results/dw4/<job-id>/`.

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
