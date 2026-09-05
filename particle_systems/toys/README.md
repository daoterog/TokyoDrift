# Toy distribution controls

This folder keeps the diagnostic experiments separate from the DW4/LJ13
particle-system benchmark:

- `figure_eight.py`: connected ring support and a depleted-crossing variant;
- `gmm40.py`: equal and imbalanced 40-mode Gaussian mixtures;
- `gmm40_report.py`: visual audit of a completed GMM-40 run;
- `notebooks/`: companion notebooks and exploratory visualisations.

Run the GMM-40 benchmark with:

```bash
uv run --project particle_systems --no-sync python -m particle_systems.toys.gmm40 \
  --distribution imbalanced \
  --output particle_systems/artifacts/runs/gmm40/imbalanced-seed42
```

The toy code does not share model or data dependencies with DW4 beyond Torch.
