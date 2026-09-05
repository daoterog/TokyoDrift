import torch
from geom_qm9.drift import unnormalised_drift
from geom_qm9.metrics import ensemble_metrics, geometry_validity
from geom_qm9.model import ConformerGenerator


def test_graph_generator_and_drift_are_equivariant_shapes():
    torch.manual_seed(0)
    model = ConformerGenerator(hidden_dim=16, layers=2, radial_basis=4)
    atom_numbers = torch.tensor([6, 6, 8])
    bond_index = torch.tensor([[0, 1], [1, 2]])
    bond_order = torch.tensor([1.0, 1.0])
    noise = torch.randn(5, 3, 3)
    generated = model(noise, atom_numbers, bond_index, bond_order)
    field, _ = unnormalised_drift(generated, generated[:3], bandwidth=1.0, repulsion=1.0)
    assert generated.shape == field.shape == (5, 3, 3)
    assert torch.allclose(generated.mean(1), torch.zeros(5, 3), atol=1e-6)


def test_conformer_metrics_are_finite():
    positions = torch.randn(6, 4, 3)
    positions -= positions.mean(1, keepdim=True)
    bonds = torch.tensor([[0, 1, 2], [1, 2, 3]])
    metrics = ensemble_metrics(positions[:3], positions[3:])
    metrics.update(geometry_validity(positions, bonds))
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
