# Data

Each benchmark owns its configuration and local dataset directory:

- `dw4/` and `lj13/` are the primary benchmarks.
- `lj55/` is retained as the larger Lennard-Jones benchmark.
- `alanine_dipeptide/` contains its fixed-topology data preparation and metrics.

Prepared datasets and source downloads are local-only and ignored by Git. Prepare a particle
dataset with `python -m data.prepare dw4`, `lj13`, or `lj55`. The default output is
`data/<system>/dataset.npz`. Prepare alanine with
`python -m data.alanine_dipeptide.prepare`.

On a compute node, run `jobs/download_env.sh` and then `jobs/download_datasets.sh`.
