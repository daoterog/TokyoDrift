# Equi

Requires Python >=3.11,<3.13 and [uv](https://docs.astral.sh/uv/).

## Install

torch and the compiled PyG packages (`pyg-lib`, `torch-scatter`, `torch-sparse`,
`torch-cluster`, `torch-spline-conv`) live in two conflicting dependency groups,
`cuda` and `cpu`. Both hold the same packages at the same versions; only the
wheel index differs. Pick one at install time:

**Linux / Windows with an NVIDIA GPU** — cu128 wheels, the default:

```bash
uv sync
```

**macOS** — must use the `cpu` group, because the cu128 index has no macOS
wheels and a plain `uv sync` will fail:

```bash
uv sync --no-group cuda --group cpu
```

torch comes from PyPI here, which on macOS is already the CPU/MPS build. The
PyG packages come from the `pyg-cpu` flat index, the only place their macOS
arm64 wheels are published.

**Linux / Windows without a GPU** — same command:

```bash
uv sync --no-group cuda --group cpu
```

Note that on Linux x86_64 this is not a slim install: PyPI's torch wheel pulls
the `nvidia-*` CUDA runtime packages regardless. The group avoids the cu128
index, not the CUDA dependencies.

### torch-linear-assignment

This is the one package with no wheels: it compiles a torch C++/CUDA extension
from source. Its own build requirement is an unpinned `torch>=1.12.0`, so an
isolated build would pull the latest PyPI torch and compile against the wrong
CUDA version. `no-build-isolation-package` in `pyproject.toml` makes it build
against the torch installed here instead. No extra flags are needed — a plain
`uv sync` handles it, and the build takes about 90 seconds.

Building the CUDA extension needs a working `nvcc` whose version matches the
torch build (12.8 for the `cuda` group). To skip CUDA and build the CPU kernel
instead, set `TLA_BUILD_CUDA=0`.

## Notes

- Which group you get cannot be decided inside `pyproject.toml` — a lockfile is
  hardware-agnostic and markers cannot see whether a GPU exists. The caller
  chooses.
- torch is pinned to 2.9.0 because the PyG wheels are built against one exact
  torch version. Before bumping it, check <https://data.pyg.org/whl/> for a
  matching index and update both flat indexes in `pyproject.toml`.
