# GEOM-QM9 data

`geom_qm9` expects three local files, deliberately excluded from Git:

```text
geom_qm9/data/processed/
  train.pt
  validation.pt
  test.pt
```

Each is a `torch.save({"molecules": records})` payload. A record contains:

```python
{
  "identifier": "optional-smiles-or-id",
  "atomic_numbers": LongTensor[n_atoms],
  "bond_index": LongTensor[2, n_bonds],       # each undirected bond once
  "bond_order": FloatTensor[n_bonds],
  "conformers": FloatTensor[n_conformers, n_atoms, 3],  # heavy atoms, Angstrom
}
```

The GEOM project publishes both raw and featurized MessagePack downloads:
<https://github.com/learningmatter-mit/geom>. Use its reader (and RDKit only
at conversion time) to export the records above, then validate each split:

```bash
uv run --project geom_qm9 python -m geom_qm9.prepare \
  --input /path/to/qm9_train_export.pt --split train --output geom_qm9/data/processed
```

Do the same for `validation` and `test`. The intended initial protocol is the
standard conformer-generation filtered split: retain molecules with at least
two conformers, keep at most 50 conformers per molecule during training, and
split by molecule—not by conformer—so test graphs are unseen.

Raw GEOM data are too large to vendor in this repository. Keeping this small,
explicit processed contract makes dataset-version and split choices auditable.
