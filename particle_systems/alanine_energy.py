"""Physical AMBER ff96/OBC1 energies for the FAB alanine dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .alanine import ATOM_NAMES, ELEMENTS, TEMPERATURE_K, TOPOLOGY_SHA256
from .prepare_data import sha256


class AlanineEnergy:
    """Evaluate unconstrained configurations, with explicit angstrom-to-nm conversion."""

    def __init__(self, topology: Path, platform: str = "Reference") -> None:
        """Construct the exact published ff96/OBC1 system without openmmtools."""
        import openmm
        from openmm import app

        if sha256(topology) != TOPOLOGY_SHA256:
            raise ValueError("energy topology checksum does not match the benchmark")
        prmtop = app.AmberPrmtopFile(str(topology))
        atoms = list(prmtop.topology.atoms())
        if (
            tuple(atom.name for atom in atoms) != ATOM_NAMES
            or tuple(atom.element.symbol for atom in atoms) != ELEMENTS
        ):
            raise ValueError("energy topology atom order is incorrect")
        system = prmtop.createSystem(
            nonbondedMethod=app.NoCutoff,
            implicitSolvent=app.OBC1,
            constraints=None,
        )
        self.integrator = openmm.VerletIntegrator(0.001)
        self.context = openmm.Context(
            system, self.integrator, openmm.Platform.getPlatformByName(platform)
        )
        self.platform = platform
        self.version = openmm.__version__

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """Return U/(RT) at 300 K; invalid energies remain NaN, never clipped."""
        import openmm
        from openmm import unit

        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim != 3 or positions.shape[1:] != (22, 3):
            raise ValueError("energy positions must have shape [batch, 22, 3]")
        values = np.full(len(positions), np.nan, dtype=np.float64)
        rt = (unit.MOLAR_GAS_CONSTANT_R * TEMPERATURE_K * unit.kelvin).value_in_unit(
            unit.kilojoule_per_mole
        )
        for i, xyz in enumerate(positions):
            if not np.isfinite(xyz).all():
                continue
            try:
                self.context.setPositions(xyz * 0.1 * unit.nanometer)
                energy = self.context.getState(getEnergy=True).getPotentialEnergy()
                value = energy.value_in_unit(unit.kilojoule_per_mole) / rt
                if np.isfinite(value):
                    values[i] = value
            except openmm.OpenMMException:
                # Degenerate coordinates can make GB/nonbonded energies undefined.
                continue
        return values
