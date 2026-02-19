from __future__ import annotations

import torch
from torch import Tensor

INFINITY = 987654.0


def _as_pairwise(
    proposals: Tensor,
    targets: Tensor,
    num_pro: int | None,
    num_tar: int | None,
) -> tuple[Tensor, Tensor]:
    if proposals.ndim != 2 or targets.ndim != 2:
        raise ValueError("proposals/targets 必须是二维张量")
    if num_pro is not None and num_tar is not None:
        expected = int(num_pro) * int(num_tar)
        if proposals.shape[0] == expected and targets.shape[0] == expected:
            return proposals.reshape(int(num_pro), int(num_tar), -1), targets.reshape(int(num_pro), int(num_tar), -1)
    return proposals[:, None, :], targets[None, :, :]


def _finalize_distance(distance: Tensor, target_vis: Tensor) -> Tensor:
    lengths = target_vis.sum(dim=-1)
    out = (distance * target_vis).sum(dim=-1) / (lengths + 1e-9)
    invalid = lengths <= 0
    if invalid.shape != out.shape:
        invalid = invalid.expand_as(out)
    out = out.masked_fill(invalid, INFINITY)
    return out


def euclidean_distance(
    proposals: Tensor,
    targets: Tensor,
    num_pro: int | None = None,
    num_tar: int | None = None,
    anchor_len: int = 10,
) -> Tensor:
    p, t = _as_pairwise(proposals, targets, num_pro, num_tar)
    target_vis = t[..., 5 + anchor_len * 2:5 + anchor_len * 3]
    distance_x = p[..., 5:5 + anchor_len] - t[..., 5:5 + anchor_len]
    distance_z = p[..., 5 + anchor_len:5 + anchor_len * 2] - t[..., 5 + anchor_len:5 + anchor_len * 2]
    pair_distance = torch.sqrt(distance_x.square() + distance_z.square())
    return _finalize_distance(pair_distance, target_vis)


def partial_euclidean_distance(
    proposals: Tensor,
    targets: Tensor,
    num_pro: int | None = None,
    num_tar: int | None = None,
    anchor_len: int = 10,
    close_weight: float = 0.7,
) -> Tensor:
    p, t = _as_pairwise(proposals, targets, num_pro, num_tar)
    target_vis = t[..., 5 + anchor_len * 2:5 + anchor_len * 3]
    distance_x = (p[..., 5:5 + anchor_len] - t[..., 5:5 + anchor_len]).abs()
    distance_z = (p[..., 5 + anchor_len:5 + anchor_len * 2] - t[..., 5 + anchor_len:5 + anchor_len * 2]).abs()
    split_idx = anchor_len // 2
    weights = p.new_full((anchor_len,), float(1.0 - close_weight))
    weights[:split_idx] = float(close_weight)
    pair_distance = torch.sqrt((distance_x * weights).square() + (distance_z * weights).square())
    return _finalize_distance(pair_distance, target_vis)


def manhattan_distance(
    proposals: Tensor,
    targets: Tensor,
    num_pro: int | None = None,
    num_tar: int | None = None,
    anchor_len: int = 10,
) -> Tensor:
    p, t = _as_pairwise(proposals, targets, num_pro, num_tar)
    target_vis = t[..., 5 + anchor_len * 2:5 + anchor_len * 3]
    distance_x = (p[..., 5:5 + anchor_len] - t[..., 5:5 + anchor_len]).abs()
    distance_z = (p[..., 5 + anchor_len:5 + anchor_len * 2] - t[..., 5 + anchor_len:5 + anchor_len * 2]).abs()
    pair_distance = distance_x + distance_z
    return _finalize_distance(pair_distance, target_vis)
