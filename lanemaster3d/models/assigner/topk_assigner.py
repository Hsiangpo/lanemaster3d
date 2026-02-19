from __future__ import annotations

import torch
from torch import Tensor

from .distance_metric import euclidean_distance, manhattan_distance, partial_euclidean_distance

INFINITY = 987654.0


class TopKLaneAssigner:
    """Top-k 匹配分配器。"""

    def __init__(
        self,
        pos_k: int = 30,
        neg_k: int = 100,
        anchor_len: int = 10,
        t_pos: float = INFINITY,
        t_neg: float = 0.0,
        neg_scale: float | None = None,
        metric: str = "Euclidean",
        **kwargs,
    ) -> None:
        self.pos_k = pos_k
        self.neg_k = neg_k
        self.t_neg = t_neg
        self.t_pos = t_pos
        self.neg_scale = neg_scale
        self.anchor_len = anchor_len
        self.metric = metric

    def _compute_distances(self, proposals: Tensor, valid_targets: Tensor, num_proposals: int, num_targets: int) -> Tensor:
        if self.metric == "Euclidean":
            return euclidean_distance(proposals, valid_targets, anchor_len=self.anchor_len)
        if self.metric == "Manhattan":
            return manhattan_distance(proposals, valid_targets, anchor_len=self.anchor_len)
        if self.metric == "Partial_Euclidean":
            return partial_euclidean_distance(proposals, valid_targets, anchor_len=self.anchor_len)
        raise ValueError(f"不支持的距离类型: {self.metric}")

    def _filter_negatives(self, distances: Tensor, positives: Tensor, proposal_distances: Tensor) -> Tensor:
        all_neg_indices = (~positives).nonzero().view(-1)
        if self.neg_scale is not None:
            t_neg = proposal_distances[positives].max() * self.neg_scale if positives.any() else self.t_neg
        else:
            t_neg = self.t_neg
        all_neg_indices = all_neg_indices[proposal_distances[~positives] > t_neg]
        perm = torch.randperm(all_neg_indices.shape[0], device=all_neg_indices.device)
        neg_k = min(self.neg_k, len(all_neg_indices))
        negative_indices = all_neg_indices[perm[:neg_k]]
        negatives = distances.new_zeros(distances.shape[0]).to(torch.bool)
        negatives[negative_indices] = True
        return negatives

    def match_proposals_with_targets(
        self,
        proposals: Tensor,
        targets: Tensor,
        return_dis: bool = False,
        **kwargs,
    ):
        valid_targets = targets[targets[:, 1] > 0]
        num_proposals = proposals.shape[0]
        num_targets = valid_targets.shape[0]
        distances = self._compute_distances(proposals, valid_targets, num_proposals, num_targets)
        min_indices = distances.min(dim=1)[1]
        row_inds = torch.arange(num_proposals, device=distances.device)
        invalid_mask = distances.new_ones(num_proposals, num_targets).to(torch.bool)
        invalid_mask[row_inds, min_indices] = False
        distances[invalid_mask] = INFINITY
        proposal_distances = distances.min(dim=1)[0]
        topk_distances, topk_indices = distances.topk(self.pos_k, dim=0, largest=False)
        all_pos_indices = topk_indices.reshape(-1)
        all_pos_distances = topk_distances.reshape(-1)
        all_pos_indices = all_pos_indices[all_pos_distances < self.t_pos]
        positives = distances.new_zeros(num_proposals).to(torch.bool)
        positives[all_pos_indices] = True
        negatives = self._filter_negatives(distances, positives, proposal_distances)
        if positives.sum() == 0:
            target_pos_idx = torch.tensor([], device=positives.device, dtype=torch.long)
            target_pos_dis = torch.tensor([], device=positives.device, dtype=torch.float32)
        else:
            target_pos_dis, target_pos_idx = distances[positives].min(dim=1)
        if return_dis:
            return positives, negatives, target_pos_idx, target_pos_dis
        return positives, negatives, target_pos_idx
