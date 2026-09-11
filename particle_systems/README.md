# Particle-system drift benchmarks

A self-contained implementation of coordinate- or descriptor-based, unnormalized Gaussian- or
Laplacian-kernel drift on the DW4, LJ13, and LJ55 particle systems from *Equivariant Flow
Matching* (Klein, Krämer, and Noé, 2023). It deliberately excludes QM9 and all
chemistry-specific code.

The method uses data attraction minus generated-sample repulsion without dividing by local
kernel mass. The generator remains E(n)-equivariant and maps centered noise directly to a
particle configuration. With `drift.descriptors: true`, the kernel compares invariant sorted
pair distances and its gradient is pulled back to particle coordinates. With `false` (also the
default for older configs), it compares flattened coordinates, which depend on particle ordering
and global orientation. The three baseline `gaussian.json` files enable descriptors.

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
nonempty list of positive numbers. With a list, the raw attraction-minus-repulsion kernel fields
are averaged with equal weight. Bandwidth schedules, when present, multiply every value in the
list by the same epoch-dependent scale. Set `training.ema_decay` to `null` to train, validate,
checkpoint, and evaluate without EMA weights; `1.0` is rejected because it would freeze EMA at
initialization.
Gaussian and Laplacian kernels omit their bandwidth-dependent probability-density prefactors;
their integrals therefore depend on bandwidth.
The `drift.normalized: false` setting separately means that the density gradient is not divided by
the local KDE mass (that is, the drift is not converted into a score).
Set `drift.kernel` to either `"gaussian"` or `"laplacian"`. Both kernels share the same
scalar/list bandwidth, raw field averaging, attraction/repulsion, logging, and checkpoint
behavior.

### Descriptor comparison and bandwidths

Set the boolean inside the `drift` object in `configs/<system>/gaussian.json`:

```json
"drift": {
  "space": "particle_coordinates",
  "kernel": "gaussian",
  "normalized": false,
  "descriptors": true
}
```

`space` describes the coordinate updates returned by the drift; `descriptors` selects the kernel
comparison. The descriptor is the sorted list of all unique interparticle distances, divided by
the square root of the number of pairs. Thus, descriptor distance is the **RMS difference of
sorted pair distances**, and bandwidth has distance units without growing merely because the
system has more pairs. No sample-dependent rescaling removes physical size information.

This choice removes translations, rotations, reflections and permutations of identical
particles. It also retains every pair distance used in the DW4 and LJ energy sums, including the
short-distance contacts that an energy scalar, radius alone, or coarse histogram would obscure.
The descriptor kernel gradient is mapped back with the descriptor Jacobian transpose, so both
attraction and repulsion produce coordinate updates and preserve the center. Reference samples
remain detached. Kernels retain their existing unnormalized, equal-weight bandwidth averaging.

Sorted distance lists are not a complete representation of geometry: different structures can
have identical lists, and pair connectivity is lost. They compare configurations within each
fixed-size benchmark; they are not a cross-species or variable-particle-count representation.
Sorting is differentiable almost everywhere, with a selected subgradient at ties; exact coincident
particles have no well-defined separating direction. These limitations mean descriptor matching
alone does not establish correct full-configuration sampling. The relevance of symmetry is
discussed in [Equivariant Flow Matching](https://arxiv.org/abs/2306.15030), and the general issue of
nonunique structural descriptors in [Pozdnyakov et al.](https://arxiv.org/abs/2001.11696).

The measured Gaussian starting bandwidths are:

| System | Data distance median | Initial generator-to-data median | Configured bandwidths |
| --- | ---: | ---: | --- |
| DW4 | 0.9173 | 1.3907 | `[0.15, 0.45, 0.9, 1.8]` |
| LJ13 | 0.1351 | 0.7178 | `[0.07, 0.14, 0.28, 0.5]` |
| LJ55 | 0.0440 | 0.2024 | `[0.022, 0.044, 0.088, 0.15]` |

These are coverage-based starting values, not optimized training results. Narrow components
resolve nearby structures; broad components provide attraction from the initial generator.
The baseline lists use a constant scale, keeping the broad component available throughout
training. `"auto"` estimates a single median in whichever comparison space is selected; it
does not check initial generator coverage. When switching to `descriptors: false`, also restore
coordinate bandwidths or use `"auto"`; the two spaces have very different distance scales.
Resume requires the same descriptor setting as the checkpoint; start a new run/output directory
when changing this setting.

See the [bandwidth audit](reports/descriptor_bandwidths/README.md) for methodology and limitations.
Reproduce the diagnostic against the local training split and configured model initialization:

```bash
uv run --project particle_systems --no-sync python -m particle_systems.check_bandwidths \
  --config particle_systems/configs/lj13/gaussian.json \
  --output /tmp/lj13-descriptor-bandwidths.json --samples 512 --seed 42
```

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

Evaluation uses every generated configuration for energy and validity statistics, scanning pair
distances in bounded batches where necessary. Quantile-based metrics and plots deterministically
sample whole configurations before expanding at most 1,000,000 pair-distance observations per
distribution, so systems with many particle pairs (such as LJ13 and LJ55) do not exceed PyTorch's
quantile size limit. The cap and sampling method are recorded in `metrics.json`; override the cap
with `--metric-sample-size` when needed.

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
