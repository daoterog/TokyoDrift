# GEOM-QM9: one-shot graph-conditioned drift

This folder is an isolated first real-generation task for the clean Gaussian
unnormalised-drift method. Given a molecular graph and Gaussian coordinate
noise, it generates a conformer in one equivariant forward pass:

\[
  (G, z) \mapsto X \sim p(X \mid G).
\]

It deliberately contains no alignment, energy feature, tail reweighting,
resampling, broad kernel, or staged sampler. Translation and rotation are
handled by the E(3)-equivariant EGNN; the drift kernel acts on sorted pair
distances within the conformer ensemble of one fixed graph.

## Dataset

Follow [`data/README.md`](data/README.md) to create the three processed split
files. GEOM must be split by molecule, never by individual conformer.

## Run

```bash
uv sync --project geom_qm9
uv run --project geom_qm9 python -m geom_qm9.train \
  --config geom_qm9/configs/gaussian.json
```

For an initial smoke test, override the steps:

```bash
uv run --project geom_qm9 python -m geom_qm9.train \
  --config geom_qm9/configs/gaussian.json --steps 10 \
  --output geom_qm9/artifacts/runs/smoke
```

Evaluate the EMA checkpoint on unseen molecular graphs:

```bash
uv run --project geom_qm9 python -m geom_qm9.evaluate \
  --checkpoint geom_qm9/artifacts/runs/gaussian-seed42/latest.pt \
  --split validation --output geom_qm9/artifacts/evaluations/gaussian-validation.json
```

The evaluator reports an alignment-free distance-matrix COV/MAT proxy plus
no-clash fraction and bond-length statistics. Standard publication COV/MAT
should later be added with RDKit symmetry-aware RMSD; it belongs solely in
evaluation, never in the training objective.
