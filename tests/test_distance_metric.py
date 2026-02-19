from __future__ import annotations

import torch

from lanemaster3d.models.assigner.distance_metric import INFINITY, euclidean_distance, manhattan_distance, partial_euclidean_distance


def _zero_vis_case(anchor_len: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    feat_dim = 5 + anchor_len * 3
    proposals = torch.zeros(1, feat_dim, dtype=torch.float32)
    targets = torch.zeros(1, feat_dim, dtype=torch.float32)
    return proposals, targets


def test_euclidean_distance_sets_infinity_for_zero_visibility() -> None:
    proposals, targets = _zero_vis_case()
    dist = euclidean_distance(proposals, targets, num_pro=1, num_tar=1, anchor_len=10)
    assert float(dist[0, 0].item()) == float(INFINITY)


def test_partial_euclidean_distance_sets_infinity_for_zero_visibility() -> None:
    proposals, targets = _zero_vis_case()
    dist = partial_euclidean_distance(proposals, targets, num_pro=1, num_tar=1, anchor_len=10)
    assert float(dist[0, 0].item()) == float(INFINITY)


def test_manhattan_distance_sets_infinity_for_zero_visibility() -> None:
    proposals, targets = _zero_vis_case()
    dist = manhattan_distance(proposals, targets, num_pro=1, num_tar=1, anchor_len=10)
    assert float(dist[0, 0].item()) == float(INFINITY)


def test_euclidean_distance_supports_pairwise_matrix_inputs() -> None:
    anchor_len = 2
    feat_dim = 5 + anchor_len * 3
    proposals = torch.zeros(2, feat_dim, dtype=torch.float32)
    targets = torch.zeros(3, feat_dim, dtype=torch.float32)
    proposals[:, 5:7] = torch.tensor([[0.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    targets[:, 5:7] = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=torch.float32)
    targets[:, 9:11] = 1.0
    dist = euclidean_distance(proposals, targets, anchor_len=anchor_len)
    assert tuple(dist.shape) == (2, 3)
    assert torch.isfinite(dist).all()
