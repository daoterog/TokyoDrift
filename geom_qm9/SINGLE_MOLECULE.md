# Single-molecule capacity diagnostic

Run from the repository root with the GEOM environment:

```bash
uv run --project geom_qm9 python -m geom_qm9.single_molecule
```

The experiment starts from scratch on `CCCCOC(C)C` (8 heavy atoms,
4 rotatable bonds, 50 conformers). It uses the existing 6.75M-parameter
one-shot model and unnormalised Gaussian drift, with the same batch size,
learning rate and bandwidth multiplier as the stable multi-molecule run.
Only the selected molecule and corrected input graph change.

The audit verifies atom/bond labels, reference bond lengths and clashes,
reference self-RMSD, rotation equivariance, nonzero initial kernel support,
and finite-difference agreement for attraction and repulsion potentials.
It saves `audit.json` alongside the run. Evaluation compares 100 generated
conformers to all 50 **training** conformers at steps 1k, 5k and 10k, with a
fixed sampling seed. This measures ability to fit one distribution; it is
not held-out molecular or conformer generalization. The reference-half
comparison is a descriptive measure of ensemble diversity, not a target
score for a model trained on all conformers.

The runner stops on any audit, training or evaluation failure. Checkpoints
include optimizer and RNG state. It refuses to overwrite existing training
checkpoints. Use the trainer's `--resume` explicitly after interruption.

## Preprocessing correction discovered by this audit

`prepare_geom.labelled_graph` previously called
`source.GetSubstructMatch(molecule)` but interpreted the result as a mapping
from source atoms to XYZ atoms. RDKit returns query-to-target indices, so the
arguments were reversed. The selected old processed record had six bonds
instead of seven, leaving a heavy atom disconnected. The corrected call is
`molecule.GetSubstructMatch(source)`.

This run rebuilds its single record from the raw archive in a separate data
directory. Existing processed datasets and historical checkpoints have not
been overwritten. Their graph labels must be audited/rebuilt before drawing
conclusions about method quality or launching further multi-molecule runs.
The corrected mapper also passed five randomly permuted XYZ-order checks.
