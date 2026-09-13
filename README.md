# TokyoDrift

Equivariant particle-generator experiments with coordinate- or descriptor-based Gaussian and
Laplacian kernel drift. DW4 and LJ13 are the primary benchmarks; LJ55 and fixed-topology alanine
dipeptide are retained as additional datasets.

## Layout

```text
.
├── train.py                   # training entry point
├── drifting.py                # coordinate-space drift implementation
├── validation.py              # deterministic checkpoint validation
├── evaluate.py                # DW4/LJ13/LJ55 evaluation
├── evaluate_alanine.py        # alanine-specific evaluation
├── data/
│   ├── dw4/                   # DW4 configs and local dataset
│   ├── lj13/                  # LJ13 configs and local dataset
│   ├── lj55/                  # retained LJ55 config and local dataset
│   ├── alanine_dipeptide/     # config, preparation, geometry, energy, and metrics
│   ├── prepare.py             # particle dataset downloader/preparation
│   └── systems.py             # benchmark definitions and potentials
├── models/                    # generator architectures
├── utils/                     # descriptors, KDE, drift diagnostics, I/O, runtime checks
├── jobs/                      # setup, download, and Slurm jobs
├── tests/                     # unit tests
└── unsure/                    # staging area for files needing later classification
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

Alanine preparation and energy evaluation additionally require:

```bash
uv sync --group alanine
```

Platform helpers and compute-node jobs live in `jobs/`.

## Prepare data

Prepare the main benchmarks independently. Each command writes a deterministic train/test archive
to its own dataset folder:

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

```bash
uv run --no-sync python -m train --config data/dw4/gaussian.json
uv run --no-sync python -m train --config data/lj13/gaussian.json
```

Every run must use its own output directory. Configurations support scalar, `"auto"`, or
multi-scale bandwidths; Gaussian or Laplacian kernels; optional invariant sorted-pair-distance
descriptors; and optional local kernel-mass normalization.

With descriptor drift, the comparison is invariant to translations, rotations, reflections, and
permutations of identical particles. The sorted distance representation is not a complete
description of geometry, so descriptor matching alone does not establish full-configuration
sampling quality. The diagnostic helper lives at `utils/check_bandwidths.py`; its recorded reports
live with the datasets under `data/reports/descriptor_bandwidths/`.

Run a diagnostic with:

```bash
uv run --no-sync python -m utils.check_bandwidths \
  --config data/lj13/gaussian.json \
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

The Slurm jobs in `jobs/` keep each run self-contained under `results/<dataset>/<run-id>/`.
Training writes into its `checkpoints/` child; evaluation writes metrics, arrays, plots, and the
combined job log into the run directory.

```text
results/<dataset>/<run-id>/
├── checkpoints/               # periodic, latest, selected, and final checkpoints
├── metrics.json
├── distributions.png          # particle benchmarks
├── energy_all_samples.png     # particle benchmarks
├── energy_valid_samples.png   # particle benchmarks
├── ramachandran.png           # alanine
├── free_energy.png            # alanine
├── slurm.out                  # scheduler output when submitted with Slurm
└── train_and_evaluate.log     # combined training and evaluation log
```

## Test

```bash
uv run --no-sync python -m unittest discover -s tests -v
```
