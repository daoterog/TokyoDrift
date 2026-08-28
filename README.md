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
four `torch-*` PyG extensions come from the `pyg-cpu` flat index. `pyg-lib` is
omitted because no macOS wheel is published for torch 2.9; it is an optional
PyG acceleration package.

**Linux / Windows without a GPU** — same command:

```bash
uv sync --no-group cuda --group cpu
```

Note that on Linux x86_64 this is not a slim install: PyPI's torch wheel pulls
the `nvidia-*` CUDA runtime packages regardless. The group avoids the cu128
index, not the CUDA dependencies.

## Notes

- Which group you get cannot be decided inside `pyproject.toml` — a lockfile is
  hardware-agnostic and markers cannot see whether a GPU exists. The caller
  chooses.
- torch is pinned to 2.9.0 because the PyG wheels are built against one exact
  torch version. Before bumping it, check <https://data.pyg.org/whl/> for a
  matching index and update both flat indexes in `pyproject.toml`.
