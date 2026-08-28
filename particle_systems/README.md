# Unnormalized drift on DW4 and LJ13

This folder is a contained particle-only adaptation of the imported unnormalized
Tokyo Drift experiment. It has no QM9, atom-type, bond, alignment, normalized
drift, or chemistry code. The training target is exactly

```text
stopgrad(generated + eta * (raw KDE attraction - repulsion * raw KDE self-repulsion))
```

The Gaussian weights are empirical expectations and are **not** divided by
their kernel mass. Reference configurations are used in their existing frames
and particle orders: there is no Kabsch or permutation alignment.

## What matches the paper

The systems and potentials follow *Equivariant flow matching* (Klein, Krämer,
Noé, NeurIPS 2023):

- DW4: four particles in 2D, `a=0`, `b=-4`, `c=0.9`, `d0=4`, `tau=1`.
- LJ13: thirteen particles in 3D, `rm=epsilon=tau=1`.
- The preparation command downloads the paper's official OSF arrays, verifies
  their SHA-256 hashes, centers every configuration, and creates deterministic,
  disjoint train/test subsets.
- Evaluation defaults to 500,000 generated samples, as used for DW4/LJ13 ESS
  in the paper. Reported repeat summaries use sample standard deviation; run
  three seeds to match the paper's error-bar convention.
- The paper's Table 1 means are embedded in each `metrics.json` for context.

There is one unavoidable non-comparability: this direct generator is not an
invertible continuous normalizing flow. It therefore has no exact proposal
density, NLL, importance weights/ESS, or integrated CNF path length. Those
fields are reported as `null`, never approximated or silently replaced.
Comparison uses quantities both methods can support without changing the
unnormalized approach: energy and pair-distance distributions, radius,
collision rate, and endpoint transport distance. In particular, energy
Wasserstein-1 and energy-histogram Jensen-Shannon divergence are zero only for
matching generated and held-out distributions.

## 1. Prepare official data

First create the isolated environment from the repository root. It deliberately
does not install any QM9, RDKit, PyG, or `torch-linear-assignment` dependency.

**macOS (Terminal):**

```bash
./particle_systems/scripts/setup_macos.sh
```

This installs the PyPI CPU/MPS build of PyTorch and verifies an EGNN
forward/backward pass plus a DW4 potential operation on MPS when available,
otherwise CPU.

**Windows with NVIDIA CUDA 12.8 (PowerShell):**

```powershell
.\particle_systems\scripts\setup_windows_cuda.ps1
```

This installs the official PyTorch CUDA 12.8 wheel and fails immediately with
an actionable error if PyTorch cannot see the NVIDIA device. A current NVIDIA
driver is required; a separately installed CUDA toolkit is not required by the
prebuilt wheel.

The equivalent setup commands are:

```bash
# macOS / CPU
uv sync --project particle_systems

# Windows / CUDA 12.8
uv sync --project particle_systems --no-group cpu --group cuda
```

Then prepare the datasets. These commands are identical in Terminal and
PowerShell because they contain no shell-specific continuations:

```bash
# 57 MB source; creates a compact 100k/100k split.
uv run --project particle_systems --no-sync python -m particle_systems.prepare_data dw4

# 1.56 GB source; it is memory-mapped while selecting the compact split.
uv run --project particle_systems --no-sync python -m particle_systems.prepare_data lj13
```

The original downloads remain under `particle_systems/data/downloads/` so a
different split can be produced without downloading again. Override sizes with
`--train-size` and `--test-size`. You may instead pass `--source /path/file.npy`.

## 2. Train

```bash
uv run --project particle_systems --no-sync python -m particle_systems.train \
  --config particle_systems/configs/dw4.json

uv run --project particle_systems --no-sync python -m particle_systems.train \
  --config particle_systems/configs/lj13.json
```

PowerShell uses a backtick instead of a backslash for multiline commands:

```powershell
uv run --project particle_systems --no-sync python -m particle_systems.train `
  --config particle_systems/configs/lj13.json --device cuda
```

Both configurations use an E(n)-equivariant direct generator, automatic median
bandwidth selection, raw detached Euler targets, and EMA checkpoints. For a
quick plumbing check, append `--steps 10`; that run is not a benchmark.

## 3. Evaluate

```bash
uv run --project particle_systems --no-sync python -m particle_systems.evaluate \
  --checkpoint particle_systems/runs/dw4/latest.pt \
  --output particle_systems/results/dw4-seed42 \
  --num-samples 500000
```

The output contains `metrics.json` and `distributions.png`. Add
`--save-samples` only when the generated coordinate archive is useful.

For paper-style uncertainty, train/evaluate seeds 42, 43, and 44 in separate
output folders (use `--seed 43 --output particle_systems/runs/dw4-seed43`, for
example), then aggregate them:

```bash
uv run --project particle_systems --no-sync python -m particle_systems.summarize \
  particle_systems/results/dw4-seed*/metrics.json \
  --output particle_systems/results/dw4-summary.json
```

## Test

```bash
uv run --project particle_systems --no-sync python -m unittest discover \
  -s particle_systems/tests -v
```

References: [paper](https://arxiv.org/abs/2306.15030),
[official datasets](https://osf.io/srqg7/).
