# TokyoDrift

Particle-generator experiments with equivariant or standard graph neural networks and
coordinate- or descriptor-based Gaussian and Laplacian kernel drift. DW4 and LJ13 are the primary
benchmarks; LJ55 and fixed-topology alanine dipeptide are retained as additional datasets.

## Layout

```text
.
├── train.py                   # training entry point
├── drifting.py                # coordinate-space drift implementation
├── validation.py              # deterministic checkpoint validation
├── evaluate.py                # DW4/LJ13/LJ55 evaluation
├── evaluate_alanine.py        # alanine-specific evaluation
├── configs/                   # dataset training configurations
├── data/
│   ├── dw4/                   # local DW4 dataset
│   ├── lj13/                  # local LJ13 dataset
│   ├── lj55/                  # local LJ55 dataset
│   ├── alanine_dipeptide/     # preparation, geometry, energy, metrics, and local dataset
│   ├── prepare.py             # particle dataset downloader/preparation
│   └── systems.py             # benchmark definitions and potentials
├── models/                    # generator architectures
├── utils/                     # descriptors, KDE, drift diagnostics, and I/O helpers
├── jobs/                      # setup, download, and Slurm jobs
└── tests/                     # unit tests
```

Generated checkpoints belong under `results/<dataset>/<run-id>/checkpoints/`, not `models/`;
`models/` contains source code only. Local arrays, downloads, results, and job logs are ignored by
Git.

## Setup

The project requires Python 3.11 or 3.12 and
[uv](https://docs.astral.sh/uv/). The CPU group is the default:

```bash
uv sync
```

For Linux or Windows with an NVIDIA GPU, use the CUDA group:

```bash
uv sync --no-group cpu --group cuda
```

Alanine preparation and energy evaluation additionally require the `alanine` dependency group.
For a CPU environment, use:

```bash
uv sync --group alanine
```

For a CUDA environment, preserve the CUDA selection while adding alanine dependencies:

```bash
uv sync --no-group cpu --group cuda --group alanine
```

Platform helpers and compute-node jobs live in `jobs/`.

## Prepare data

Prepare the main benchmarks independently. Each command writes deterministic train, validation,
and test splits to its own dataset folder. DW4, LJ13, and LJ55 all use the same default split sizes:
100,000 training, 400,000 validation, and 500,000 test configurations.

```bash
uv run --no-sync python -m data.prepare dw4
uv run --no-sync python -m data.prepare lj13
```

LJ55 remains available with `python -m data.prepare lj55`. Its published data consists of two
large source arrays, so allow substantial download and working space.

Prepare the official alanine train/validation/test splits with:

```bash
uv run --no-sync python -m data.alanine_dipeptide.prepare
```

The particle data job prepares DW4, LJ13, and LJ55. DW4 and LJ13 should be treated as the default
development and comparison targets.

## Train

For Slurm runs, use the submission wrapper. Each invocation immediately submits four independent
jobs using the same base config and seed: EGNN and GNN, each with normalized and unnormalized
drifting. Every queued job receives its own immutable derived snapshot:

```bash
jobs/submit_training.sh dw4
jobs/submit_training.sh lj13 --run-id experiment-17
jobs/submit_training.sh lj55 --config configs/lj55_config.json -- --seed 17
```

Snapshots are stored under `results/submitted-configs/<system>/<run-id>/` and copied to the
eventual run directories as `submitted_config.json`. Editing a config after submission therefore
does not affect queued jobs. Arguments after `--` are stored by Slurm and forwarded to all four
training commands; normalization overrides are rejected because they conflict with the matrix.
The supplied model settings define the EGNN reference capacity. For the GNN snapshots, submission
keeps `hidden_dim` and `layers` fixed, then directly reduces `radial_basis` by 3 for DW4 (2D)
or by 6 for LJ13, LJ55, and alanine (3D). This compensates for the GNN's additional coordinate
weights. The generated snapshot counts and resolved widths are printed before the jobs are
submitted.
Every training job evaluates its selected checkpoint with 500,000 generated samples.

For an immediate local run, invoke the training module directly:

```bash
uv run --no-sync python -m train --config configs/dw4_config.json \
  --output results/dw4/example-run/checkpoints
uv run --no-sync python -m train --config configs/lj13_config.json \
  --output results/lj13/example-run/checkpoints
```

Replace `example-run` with a unique run identifier. The checked-in configs contain baseline output
paths, so pass `--output` when running multiple experiments to avoid reusing an existing directory.
Set `model.architecture` to `"egnn"` for an E(n)-equivariant generator or `"gnn"` for a graph
neural network whose coordinate updates are not constrained to be E(n)-equivariant. Configurations
also support scalar, `"auto"`, or multi-scale bandwidths; Gaussian or Laplacian kernels; optional
invariant sorted-pair-distance descriptors; and optional local kernel-mass normalization.

Checked-in EGNN configs use `model.variant: "block"` ([models/egnn.py](models/egnn.py)). Each of
the `model.layers` blocks applies `model.layers_per_block` feature message-passing sublayers
(optional sigmoid attention, residual update) and then one coordinate update. Edge attributes are
the initial and current squared distances; coordinate messages are
`(x_j - x_i) / (|x_j - x_i| + 1)` times a learned weight bounded by
`tanh(·) * model.coordinate_range` (default 15) and are combined with `model.aggregation`
(`"sum"` by default, or `"mean"`). The block EGNN has no radial basis; `model.max_distance` is used
only by the derived GNN baseline, whose `radial_basis` the training matrix chooses to match the
EGNN parameter count.

The `"legacy"` and `"bounded"` variants build the earlier radial-basis EGNN
([models/egnn_legacy.py](models/egnn_legacy.py)) and require `model.radial_basis`. Configs without
a variant resolve to `"legacy"` so existing checkpoints retain their original parameter shapes and
behavior.

With descriptor drift, the comparison is invariant to translations, rotations, reflections, and
permutations of identical particles. The sorted distance representation is not a complete
description of geometry, so descriptor matching alone does not establish full-configuration
sampling quality. The diagnostic helper lives at `utils/check_bandwidths.py`; its recorded reports
live with the datasets under `data/reports/descriptor_bandwidths/`.

Run a diagnostic with:

```bash
uv run --no-sync python -m utils.check_bandwidths \
  --config configs/lj13_config.json \
  --output /tmp/lj13-descriptor-bandwidths.json --samples 512 --seed 42
```

## Evaluate

```bash
uv run --no-sync python -m evaluate \
  --checkpoint results/dw4/example/checkpoints/final.pt \
  --output results/dw4/example --num-samples 500000
```

DW4 and LJ13 evaluation includes an optional normalized Gaussian KDE score on held-out
sorted-pair-distance descriptors. It reports generated-minus-reference excess NLL alongside
energy, validity, pair-distance, and radius metrics. Use `--skip-kde-nll` when the KDE diagnostic
is not needed.

Alanine uses labeled pair distances rather than sorted identical-particle descriptors. Evaluate it
with `python -m evaluate_alanine`; validation combines periodic backbone-angle agreement with
geometry validity, and the full evaluation can add AMBER ff96/OBC1 energy metrics.

The DW4, LJ13, and LJ55 Slurm jobs keep each run self-contained under
`results/<dataset>/<run-id>_<architecture>_<norm|unnorm>_<descr|nodescr>/`. The architecture,
normalization, and descriptor labels are read from the selected JSON config. For example, run 123
with an unnormalized descriptor EGNN is stored in `123_egnn_unnorm_descr/`.
Training writes a single `checkpoints/final.pt` at the configured epoch limit or when mean epoch
training loss fails to improve by more than `early_stopping_epsilon` for
`early_stopping_patience` consecutive epochs. Validation remains diagnostic and does not select
a checkpoint. The particle jobs also record fixed-sample train/validation drift loss and energy
Wasserstein distance at each configured tracking interval in
`checkpoints/train_validation_history.jsonl`. The test split is read only by the final evaluator.

```text
results/<dataset>/<run-id>_<architecture>_<norm|unnorm>_<descr|nodescr>/
├── checkpoints/               # parameters, diagnostic histories, and final.pt
├── metrics.json
├── distributions.png          # particle benchmarks
├── energy_all_samples.png     # particle benchmarks
├── energy_valid_samples.png   # particle benchmarks
├── loss_over_epochs.png       # periodic particle train/validation drift loss
├── energy_wasserstein_over_epochs.png # periodic particle train/validation distance
├── ramachandran.png           # alanine
├── free_energy.png            # alanine
├── slurm.out                  # scheduler output when submitted with Slurm
└── train_and_evaluate.log     # combined training and evaluation log
```

## Test

```bash
uv run --no-sync python -m unittest discover -s tests -v
```
