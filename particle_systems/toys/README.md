# One-shot kernel-drifting toy experiments

This folder contains controlled low-dimensional experiments used to diagnose mode
coverage and imbalanced distributions:

- `figure_eight.py`: balanced connected figure-eight distribution;
- `gmm40.py`: equal and imbalanced 40-mode Gaussian mixtures;
- `gmm40_report.py`: plots and metrics for a completed GMM-40 run;
- `notebooks/`: notes for companion interactive visualisations.

## Environment

Requirements: Python 3.11 or 3.12 and either [uv](https://docs.astral.sh/uv/)
(recommended) or pip. The supplied `pyproject.toml` pins PyTorch and declares
all runtime dependencies.

### Install with uv

From this directory:

```bash
uv sync
```

On Apple Silicon this installs the standard PyTorch build and uses MPS when it
is available. On Linux with an NVIDIA GPU, install the PyTorch build appropriate
for the local CUDA version before running these scripts.

### Install with pip

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 matplotlib
```

## Run experiments

Run the figure-eight control:

```bash
uv run python figure_eight.py \
  --output artifacts/figure-eight/seed91
```

Run the imbalanced GMM-40 experiment:

```bash
uv run python gmm40.py \
  --distribution imbalanced \
  --output artifacts/gmm40/imbalanced-seed42
```

Create the GMM-40 visual report after training:

```bash
uv run python gmm40_report.py \
  --run artifacts/gmm40/imbalanced-seed42 \
  --distribution imbalanced
```

Replace `uv run python` with `python` after activating the pip environment.
Each training experiment writes checkpoints and `metrics.jsonl` below its output
directory. `gmm40_report.py` writes `validity_audit.png` beside the selected run.
Runs are deterministic for their configured seed.
