"""One-shot Gaussian-kernel drift in graph-labelled internal coordinates."""

from __future__ import annotations

from itertools import combinations

import torch


def angle_indices(bond_index: torch.Tensor, atoms: int) -> torch.Tensor:
    """Enumerate labelled ``left-centre-right`` graph angles once per molecule."""
    neighbours = [[] for _ in range(atoms)]
    for left, right in bond_index.T.tolist():
        neighbours[left].append(right)
        neighbours[right].append(left)
    angles = []
    for centre, values in enumerate(neighbours):
        for left, right in combinations(sorted(values), 2):
            angles.append((left, centre, right))
    return torch.tensor(angles, dtype=torch.long, device=bond_index.device).reshape(-1, 3).T


def torsion_indices(bond_index: torch.Tensor, atoms: int) -> torch.Tensor:
    """Enumerate labelled proper torsions over every graph bond once."""
    neighbours = [[] for _ in range(atoms)]
    for left, right in bond_index.T.tolist():
        neighbours[left].append(right)
        neighbours[right].append(left)
    torsions = []
    for first, second in bond_index.T.tolist():
        centre_left, centre_right = sorted((first, second))
        for left in sorted(value for value in neighbours[centre_left] if value != centre_right):
            for right in sorted(value for value in neighbours[centre_right] if value != centre_left):
                torsions.append((left, centre_left, centre_right, right))
    return torch.tensor(torsions, dtype=torch.long, device=bond_index.device).reshape(-1, 4).T


def three_hop_pair_indices(bond_index: torch.Tensor, atoms: int) -> torch.Tensor:
    """Return labelled pairs separated by three graph edges.

    These distances directly constrain the end points of torsional paths and
    act as local ring-closure features for small rings, without reverting to a
    global anonymous distance multiset.
    """
    neighbours = [[] for _ in range(atoms)]
    for left, right in bond_index.T.tolist():
        neighbours[left].append(right)
        neighbours[right].append(left)
    pairs = []
    for source in range(atoms):
        distance = {source: 0}
        frontier = [source]
        while frontier:
            parent = frontier.pop(0)
            if distance[parent] == 3:
                continue
            for child in neighbours[parent]:
                if child not in distance:
                    distance[child] = distance[parent] + 1
                    frontier.append(child)
        pairs.extend((source, target) for target, hops in distance.items() if target > source and hops == 3)
    return torch.tensor(pairs, dtype=torch.long, device=bond_index.device).reshape(-1, 2).T


def descriptor(
    positions: torch.Tensor,
    bond_index: torch.Tensor | None = None,
    angles: torch.Tensor | None = None,
    torsions: torch.Tensor | None = None,
    three_hop_pairs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return invariant geometry; graph-labelled bonds and angles when supplied.

    The no-graph fallback is retained only for the old diagnostic proxy metrics.
    Training supplies a fixed molecular graph, where every descriptor coordinate
    refers to a particular labelled bond or labelled graph angle—there is no
    sorting and therefore no loss of local correspondence.
    """
    atoms = positions.shape[1]
    if bond_index is None:
        pairs = torch.triu_indices(atoms, atoms, offset=1, device=positions.device)
        distances = (positions[:, pairs[0]] - positions[:, pairs[1]]).square().sum(-1).sqrt()
        return distances.sort(dim=-1).values
    bond_distance = (positions[:, bond_index[0]] - positions[:, bond_index[1]]).square().sum(-1).sqrt()
    if angles is None:
        angles = angle_indices(bond_index, atoms)
    components = [bond_distance]
    if angles.numel():
        left = positions[:, angles[0]] - positions[:, angles[1]]
        right = positions[:, angles[2]] - positions[:, angles[1]]
        cosine = (left * right).sum(-1) / (left.square().sum(-1).sqrt() * right.square().sum(-1).sqrt()).clamp_min(1e-8)
        components.append(torch.acos(cosine.clamp(-1.0 + 1e-6, 1.0 - 1e-6)))
    if torsions is None:
        torsions = torsion_indices(bond_index, atoms)
    if torsions.numel():
        first = positions[:, torsions[1]] - positions[:, torsions[0]]
        middle = positions[:, torsions[2]] - positions[:, torsions[1]]
        last = positions[:, torsions[3]] - positions[:, torsions[2]]
        axis = middle / middle.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-8)
        left = first - (first * axis).sum(-1, keepdim=True) * axis
        right = last - (last * axis).sum(-1, keepdim=True) * axis
        normaliser = (left.square().sum(-1).sqrt() * right.square().sum(-1).sqrt()).clamp_min(1e-8)
        components.extend(((left * right).sum(-1) / normaliser, (torch.cross(axis, left, dim=-1) * right).sum(-1) / normaliser))
    if three_hop_pairs is None:
        three_hop_pairs = three_hop_pair_indices(bond_index, atoms)
    if three_hop_pairs.numel():
        components.append((positions[:, three_hop_pairs[0]] - positions[:, three_hop_pairs[1]]).square().sum(-1).sqrt())
    return torch.cat(components, dim=-1)


@torch.no_grad()
def descriptor_statistics(
    samples: torch.Tensor,
    bond_index: torch.Tensor,
    angles: torch.Tensor,
    torsions: torch.Tensor | None = None,
    three_hop_pairs: torch.Tensor | None = None,
    maximum: int = 512,
    minimum_scale: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Whiten labelled internal coordinates using that molecule's ensemble."""
    values = descriptor(samples[:maximum], bond_index, angles, torsions, three_hop_pairs)
    mean = values.mean(0)
    # A physical floor prevents essentially rigid coordinates from producing
    # arbitrarily large Cartesian forces. It is in Å for bonds and radians for
    # angles, so it also defines the resolution the kernel asks the generator
    # to resolve at this first, local-geometry stage.
    scale = values.std(0, unbiased=False).clamp_min(minimum_scale)
    return mean, scale


@torch.no_grad()
def median_bandwidth(
    samples: torch.Tensor,
    bond_index: torch.Tensor | None = None,
    angles: torch.Tensor | None = None,
    torsions: torch.Tensor | None = None,
    three_hop_pairs: torch.Tensor | None = None,
    mean: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    maximum: int = 512,
) -> float:
    """Return median RMS separation of (optionally whitened) descriptors."""
    values = descriptor(samples[:maximum], bond_index, angles, torsions, three_hop_pairs)
    if mean is not None or scale is not None:
        if mean is None or scale is None:
            raise ValueError("descriptor mean and scale must be supplied together")
        values = (values - mean) / scale
    return float((torch.pdist(values).square() / values.shape[-1]).median().sqrt().clamp_min(1e-3))


def field(
    query: torch.Tensor,
    references: torch.Tensor,
    bandwidth: float,
    self_field: bool,
    bond_index: torch.Tensor | None = None,
    angles: torch.Tensor | None = None,
    torsions: torch.Tensor | None = None,
    three_hop_pairs: torch.Tensor | None = None,
    mean: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate an unnormalised Gaussian kernel density in Cartesian space."""
    with torch.enable_grad():
        differentiable = query.detach().requires_grad_(True)
        reference_descriptor = descriptor(references.detach(), bond_index, angles, torsions, three_hop_pairs)
        query_descriptor = descriptor(differentiable, bond_index, angles, torsions, three_hop_pairs)
        if mean is not None or scale is not None:
            if mean is None or scale is None:
                raise ValueError("descriptor mean and scale must be supplied together")
            reference_descriptor = (reference_descriptor - mean) / scale
            query_descriptor = (query_descriptor - mean) / scale
        residual = reference_descriptor[None] - query_descriptor[:, None]
        squared_error = residual.square().mean(-1)
        kernel = torch.exp(-squared_error / (2.0 * bandwidth**2))
        if self_field:
            kernel = kernel * (1.0 - torch.eye(len(query), device=query.device, dtype=query.dtype))
        density = residual.shape[-1] * kernel.mean(1)
        (value,) = torch.autograd.grad(density.sum(), differentiable)
    return value.detach(), kernel.detach().mean()


def unnormalised_drift(
    generated: torch.Tensor,
    positive: torch.Tensor,
    bandwidth: float,
    repulsion: float,
    bond_index: torch.Tensor | None = None,
    angles: torch.Tensor | None = None,
    torsions: torch.Tensor | None = None,
    three_hop_pairs: torch.Tensor | None = None,
    mean: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Attract labelled internal geometry to data and repel generated geometry."""
    attraction, positive_mass = field(
        generated, positive, bandwidth, self_field=False,
        bond_index=bond_index, angles=angles, torsions=torsions, three_hop_pairs=three_hop_pairs, mean=mean, scale=scale,
    )
    repulsive, negative_mass = field(
        generated, generated, bandwidth, self_field=True,
        bond_index=bond_index, angles=angles, torsions=torsions, three_hop_pairs=three_hop_pairs, mean=mean, scale=scale,
    )
    value = attraction - repulsion * repulsive
    value = value - value.mean(1, keepdim=True)
    return value, {
        "positive_kernel_mass": float(positive_mass),
        "negative_kernel_mass": float(negative_mass),
        "drift_rms": float(value.square().mean().sqrt()),
    }
