"""PyTorch Lightning DataModule for the DW4 and LJ13 many-particle systems.

Mirrors the structure of the QM9 DataModule: the same graph representation
(centered positions, one-hot node types, fully-connected `dense_edge_index`),
the same particle-count-balanced batching, and the same dataloader API.

The systems come from Kohler et al., "Equivariant Flows" (the bgflow paper).
bgflow itself ships no dataset file -- it generates configurations by running
MCMC against an analytic energy. The benchmark everyone actually compares
against is the fixed set of samples released with Garcia Satorras et al.,
"E(n) Equivariant Normalizing Flows", so those files are what this module
downloads, along with their train/val/test protocol.

`n_train` is the experimental variable in this benchmark, not a detail: the
published numbers sweep it (DW4 at 100/1000/10000, LJ13 at 10/100/1000) to
measure sample efficiency. Val and test are fixed at 1000 configurations.

The analytic energies are kept here even though no sampling is done with them,
because evaluating generated configurations means scoring their energy. See
`particles_eval.py`.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, download_url
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Center, Compose

from .qm9 import AtomCountBatchSampler, FullyConnectedTransform

_DATA_URL = "https://raw.githubusercontent.com/vgsatorras/en_flows/main/dw4_experiment/data"

_N_VAL = 1_000
_N_TEST = 1_000

_DISTANCE_EPS = 1e-6


class MultiDoubleWellPotential:
    """Pairwise double-well energy of a many-particle system.

    The energy of a pair at distance ``d`` is

        E(d) = a * (d - offset)^4 + b * (d - offset)^2 + c

    summed over all unique particle pairs. The defaults are the DW4 parameters.
    """

    def __init__(
        self,
        n_particles: int,
        n_dimensions: int,
        a: float = 0.9,
        b: float = -4.0,
        c: float = 0.0,
        offset: float = 4.0,
    ) -> None:
        """Initialize the double-well potential.

        Args:
            n_particles: Number of particles in the system.
            n_dimensions: Spatial dimension of each particle.
            a: Quartic coefficient.
            b: Quadratic coefficient.
            c: Constant offset of the pair energy.
            offset: Distance offset of the well.
        """
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.a = a
        self.b = b
        self.c = c
        self.offset = offset

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the energy of a batch of configurations.

        Args:
            x: Positions of shape ``[batch, n_particles, n_dimensions]``.

        Returns:
            Energies of shape ``[batch]``.
        """
        return self.pair_energy(pairwise_distances(x)).sum(dim=-1)

    def pair_energy(self, d: torch.Tensor) -> torch.Tensor:
        """Compute the energy of a single pair at distance d.

        Exposed separately so evaluation code can plot the potential and build
        the analytic reference density over distances.

        Args:
            d: Pair distances, any shape.

        Returns:
            Pair energies, same shape as `d`.
        """
        d = d - self.offset
        return self.a * d**4 + self.b * d**2 + self.c


class LennardJonesPotential:
    """Lennard-Jones cluster energy with an optional harmonic confinement.

    The pair energy is ``eps * ((rm / d)^12 - 2 * (rm / d)^6)``, summed over all
    unique pairs. The defaults are the LJ13 parameters.
    """

    def __init__(
        self,
        n_particles: int,
        n_dimensions: int,
        eps: float = 1.0,
        rm: float = 1.0,
        oscillator: bool = True,
        oscillator_scale: float = 1.0,
    ) -> None:
        """Initialize the Lennard-Jones potential.

        Args:
            n_particles: Number of particles in the system.
            n_dimensions: Spatial dimension of each particle.
            eps: LJ well depth.
            rm: LJ well radius.
            oscillator: Whether to add a harmonic oscillator confining the cluster.
            oscillator_scale: Force constant of the harmonic oscillator term.
        """
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.eps = eps
        self.rm = rm
        self.oscillator = oscillator
        self.oscillator_scale = oscillator_scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the energy of a batch of configurations.

        Args:
            x: Positions of shape ``[batch, n_particles, n_dimensions]``.

        Returns:
            Energies of shape ``[batch]``.
        """
        energies = self.pair_energy(pairwise_distances(x)).sum(dim=-1)

        if self.oscillator:
            centered = x - x.mean(dim=-2, keepdim=True)
            energies = energies + self.oscillator_scale * 0.5 * centered.pow(2).sum(
                dim=(-2, -1)
            )

        return energies

    def pair_energy(self, d: torch.Tensor) -> torch.Tensor:
        """Compute the energy of a single pair at distance d.

        Args:
            d: Pair distances, any shape.

        Returns:
            Pair energies, same shape as `d`.
        """
        ratio = self.rm / d.clamp_min(_DISTANCE_EPS)
        return self.eps * (ratio**12 - 2 * ratio**6)


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """Compute distances between all unique particle pairs.

    Args:
        x: Positions of shape ``[batch, n_particles, n_dimensions]``.

    Returns:
        Distances of shape ``[batch, n_particles * (n_particles - 1) // 2]``.
    """
    n_particles = x.shape[-2]
    rows, cols = torch.triu_indices(n_particles, n_particles, offset=1, device=x.device)
    return torch.cdist(x, x)[..., rows, cols]


@dataclass(frozen=True)
class ParticleSystem:
    """Definition of a particle system: its geometry, energy and raw files."""

    name: str
    n_particles: int
    n_dimensions: int
    energy_cls: type
    energy_kwargs: dict = field(default_factory=dict)
    raw_file_names: tuple[str, ...] = ()

    @property
    def dim(self) -> int:
        """Flat dimensionality of a configuration.

        Returns:
            ``n_particles * n_dimensions``.
        """
        return self.n_particles * self.n_dimensions

    def build_energy(self) -> MultiDoubleWellPotential | LennardJonesPotential:
        """Instantiate the energy of this system.

        Returns:
            The callable energy for this particle system.
        """
        return self.energy_cls(
            self.n_particles, self.n_dimensions, **self.energy_kwargs
        )


SYSTEMS: dict[str, ParticleSystem] = {
    # 4 particles in 2D with pairwise double-well interactions.
    "dw4": ParticleSystem(
        name="dw4",
        n_particles=4,
        n_dimensions=2,
        energy_cls=MultiDoubleWellPotential,
        energy_kwargs={"a": 0.9, "b": -4.0, "c": 0.0, "offset": 4.0},
        raw_file_names=("dw4-dataidx.npy",),
    ),
    # 13 particles in 3D, the Lennard-Jones cluster.
    "lj13": ParticleSystem(
        name="lj13",
        n_particles=13,
        n_dimensions=3,
        energy_cls=LennardJonesPotential,
        energy_kwargs={
            "eps": 1.0,
            "rm": 1.0,
            "oscillator": True,
            "oscillator_scale": 1.0,
        },
        raw_file_names=(
            "all_data_LJ13.npy",
            "holdout_data_LJ13.npy",
            "idx_LJ13.npy",
        ),
    ),
}


class EncodeParticleTypesTransform:
    """A PyG transform that converts particle type indices to one-hot vectors.

    The QM9 counterpart one-hot encodes five atomic species. These systems have a
    single species, so the encoding is a column of ones -- kept anyway so the
    batches carry the same fields and a model written against QM9 runs unchanged.
    """

    def __init__(self, n_particle_types: int = 1) -> None:
        """Initialize the transform.

        Args:
            n_particle_types: Number of distinct particle species.
        """
        self.n_particle_types = n_particle_types

    def __call__(self, data: Data) -> Data:
        """Convert particle types in data.z to one-hot vectors.

        Args:
            data: PyTorch Geometric Data object with z (type indices) attribute.

        Returns:
            Modified Data object with real_atom_types added as one-hot encoding.
        """
        data.real_atom_types = F.one_hot(
            data.z, num_classes=self.n_particle_types
        ).float()
        return data


class ParticlesDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the DW4 and LJ13 particle systems."""

    def __init__(
        self,
        root: str = "data/Particles",
        system: str = "dw4",
        n_real_molecules: int = 128,
        n_train: int = 1_000,
        num_workers: int = 0,
        force_reload: bool = False,
    ):
        """Initialize the particle DataModule.

        Args:
            root: Root directory to download and cache the raw configurations.
            system: Which system to use, one of the keys of `SYSTEMS`.
            n_real_molecules: Batch size for dataloaders. Named as in QM9DataModule
                so the two modules stay interchangeable downstream.
            n_train: Number of training configurations. The benchmark's sample
                efficiency axis; published sweeps use 100/1000/10000 for DW4 and
                10/100/1000 for LJ13.
            num_workers: Number of worker processes for loading. Defaults to 0
                because the dataset lives in memory and collation is cheap;
                a positive value pickles the whole graph list into each worker.
            force_reload: Re-download the raw files even if they are cached.

        Raises:
            ValueError: If `system` is not a known particle system.
        """
        super().__init__()
        if system not in SYSTEMS:
            raise ValueError(
                f"Unknown system {system!r}. Available: {sorted(SYSTEMS)}."
            )

        self.root = root
        self.spec = SYSTEMS[system]
        self.system = system
        self.n_real_molecules = n_real_molecules
        self.n_train = n_train
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.pin_memory = torch.cuda.is_available()

        self.n_particles = self.spec.n_particles
        self.n_dimensions = self.spec.n_dimensions
        self.target_energy = self.spec.build_energy()

    @property
    def raw_dir(self) -> Path:
        """Directory holding the downloaded .npy files.

        Returns:
            Path to the raw directory.
        """
        return Path(self.root) / self.system / "raw"

    def prepare_data(self) -> None:
        """Download the raw configuration files once, on a single process."""
        for name in self.spec.raw_file_names:
            path = self.raw_dir / name
            if path.exists() and not self.force_reload:
                continue
            download_url(f"{_DATA_URL}/{name}", str(self.raw_dir))

    def setup(self, stage: str | None = None) -> None:
        """Load the benchmark splits and build the graph dataset.

        The splits are the published ones, not a re-partition: reshuffling them
        would make the numbers incomparable with the literature.

        Args:
            stage: Lightning stage, unused.
        """
        self.prepare_data()

        if self.system == "dw4":
            splits = self._load_dw4()
        else:
            splits = self._load_lj13()

        train, val, test = (self._center(x) for x in splits)

        # One flat list plus index groups, as in QM9DataModule: the dataloaders
        # index a single dataset through a batch sampler rather than three.
        dataset = []
        offsets = {}
        for name, positions in [("train", train), ("val", val), ("test", test)]:
            offsets[name] = np.arange(len(dataset), len(dataset) + len(positions))
            dataset.extend(self._build_graphs(positions))

        self.dataset = dataset
        self.train_set = [dataset[i] for i in offsets["train"]]
        self.val_set = [dataset[i] for i in offsets["val"]]
        self.test_set = [dataset[i] for i in offsets["test"]]

        self.train_indices_by_num_atoms = self._group_by_num_atoms(offsets["train"])
        self.val_indices_by_num_atoms = self._group_by_num_atoms(offsets["val"])
        self.test_indices_by_num_atoms = self._group_by_num_atoms(offsets["test"])

        self.available_train_num_atoms = sorted(self.train_indices_by_num_atoms)
        self.available_val_num_atoms = sorted(self.val_indices_by_num_atoms)
        self.available_test_num_atoms = sorted(self.test_indices_by_num_atoms)

    def _load_dw4(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load the DW4 splits.

        The file is a pickled ``(data, idx)`` pair; ``idx`` is a reshuffling that
        the reference implementation loads and then ignores, so it is dropped
        here. The last 2000 configurations are held out as val and test.

        Returns:
            Tuple of (train, val, test) tensors of shape ``[n, dim]``.

        Raises:
            ValueError: If `n_train` exceeds the available training configurations.
        """
        data = np.load(self.raw_dir / "dw4-dataidx.npy", allow_pickle=True)[0]
        data = torch.as_tensor(data).reshape(-1, self.spec.dim)

        n_holdout = _N_VAL + _N_TEST
        train = data[: len(data) - n_holdout]
        val = data[len(data) - n_holdout : len(data) - _N_TEST]
        test = data[len(data) - _N_TEST :]

        return self._take_train(train), val, test

    def _load_lj13(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load the LJ13 splits.

        Train comes from a separate holdout file, subsampled through a fixed
        index permutation so that a smaller `n_train` is a prefix of a larger
        one. Val and test are the first 2000 rows of the main file.

        Returns:
            Tuple of (train, val, test) tensors of shape ``[n, dim]``.

        Raises:
            ValueError: If `n_train` exceeds the available training configurations.
        """
        all_data = np.load(self.raw_dir / "all_data_LJ13.npy")
        holdout = np.load(self.raw_dir / "holdout_data_LJ13.npy")
        idx = np.load(self.raw_dir / "idx_LJ13.npy")

        holdout = torch.as_tensor(holdout).reshape(-1, self.spec.dim)
        all_data = torch.as_tensor(all_data).reshape(-1, self.spec.dim)

        train = holdout[idx[: self.n_train]]
        test = all_data[:_N_TEST]
        val = all_data[_N_TEST : _N_TEST + _N_VAL]

        return self._take_train(train, already_selected=True), val, test

    def _take_train(
        self,
        train: torch.Tensor,
        already_selected: bool = False,
    ) -> torch.Tensor:
        """Cut the training pool down to `n_train` configurations.

        Args:
            train: The full training pool, or the already-selected subset.
            already_selected: True if the caller has applied `n_train` itself.

        Returns:
            Exactly `n_train` training configurations.

        Raises:
            ValueError: If `n_train` exceeds the available training configurations.
        """
        if already_selected:
            available = len(train)
        else:
            available = len(train)
            train = train[: self.n_train]

        if self.n_train > available:
            raise ValueError(
                f"n_train={self.n_train} exceeds the {available} training "
                f"configurations available for {self.system}."
            )
        return train.clone()

    def _center(self, positions: torch.Tensor) -> torch.Tensor:
        """Make each configuration mean-free.

        The reference implementation centers before splitting into graphs, and
        the models are only equivariant on the mean-free subspace, so this is
        part of the data definition rather than an augmentation.

        Args:
            positions: Configurations of shape ``[n, dim]``.

        Returns:
            Mean-free configurations of shape ``[n, n_particles, n_dimensions]``.
        """
        positions = positions.reshape(-1, self.n_particles, self.n_dimensions)
        return positions - positions.mean(dim=1, keepdim=True)

    def _build_graphs(self, positions: torch.Tensor) -> list[Data]:
        """Turn configurations into fully-connected PyG graphs.

        Args:
            positions: Positions of shape ``[n, n_particles, n_dimensions]``.

        Returns:
            List of Data objects with pos, z, real_atom_types and dense_edge_index.
        """
        transform = Compose(
            [Center(), FullyConnectedTransform(), EncodeParticleTypesTransform()]
        )
        types = torch.zeros(self.n_particles, dtype=torch.long)
        return [transform(Data(pos=pos, z=types)) for pos in positions]

    def _group_by_num_atoms(self, indices: np.ndarray) -> dict[int, list[int]]:
        """Group dataset indices by the number of particles in each configuration.

        Every configuration in a system has the same particle count, so this
        yields a single group. It is kept because it is what makes the batch
        contract identical to QM9's: every graph in a batch has equal node count.

        Args:
            indices: Array of indices to group.

        Returns:
            Dict mapping particle count to list of indices.
        """
        groups = defaultdict(list)
        for idx in indices:
            groups[self.n_particles].append(int(idx))
        return dict(groups)

    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader with particle-count-balanced batches.

        Returns:
            DataLoader with AtomCountBatchSampler.
        """
        return DataLoader(
            self.dataset,
            batch_sampler=AtomCountBatchSampler(
                self.train_indices_by_num_atoms,
                batch_size=self.n_real_molecules,
                shuffle=True,
                replacement=False,
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader with particle-count-balanced batches.

        Returns:
            DataLoader with AtomCountBatchSampler.
        """
        return DataLoader(
            self.dataset,
            batch_sampler=AtomCountBatchSampler(
                self.val_indices_by_num_atoms,
                batch_size=self.n_real_molecules,
                shuffle=False,
                replacement=False,
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test dataloader with particle-count-balanced batches.

        Returns:
            DataLoader with AtomCountBatchSampler.
        """
        return DataLoader(
            self.dataset,
            batch_sampler=AtomCountBatchSampler(
                self.test_indices_by_num_atoms,
                batch_size=self.n_real_molecules,
                shuffle=False,
                replacement=False,
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
