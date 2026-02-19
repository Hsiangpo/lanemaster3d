from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..assigner import build_assigner
from .kornia_focal import FocalLoss


class LaneSetLoss(nn.Module):
    """完整迁移版车道集合损失。"""

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        anchor_len: int = 10,
        gt_anchor_len: int = 200,
        anchor_steps: list[int] | None = None,
        use_sigmoid: bool = False,
        loss_weights: dict | None = None,
        anchor_assign: bool = True,
        assign_cfg: dict | None = None,
        strict_class_check: str = "warn",
    ) -> None:
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.anchor_len = anchor_len
        self.gt_anchor_len = gt_anchor_len
        self.anchor_steps = np.array(anchor_steps or [], dtype=np.int64) - 1
        self.use_sigmoid = use_sigmoid
        self.anchor_assign = anchor_assign
        self.loss_weights = loss_weights or {
            "cls_loss": 1.0,
            "reg_losses_x": 1.0,
            "reg_losses_z": 1.0,
            "reg_losses_vis": 1.0,
        }
        if strict_class_check not in {"off", "warn", "raise"}:
            raise ValueError("strict_class_check 仅支持 off/warn/raise")
        self.strict_class_check = strict_class_check
        self.class_index_issue_total = 0
        self._warned_class_issue = False
        self.assigner = build_assigner(assign_cfg or {})

    def _slice_target(self, target: Tensor) -> Tensor:
        x_idx = torch.tensor(self.anchor_steps, dtype=torch.long, device=target.device) + 5
        z_idx = x_idx + self.gt_anchor_len
        vis_idx = x_idx + self.gt_anchor_len * 2
        x_target = target.index_select(1, x_idx)
        z_target = target.index_select(1, z_idx)
        vis_target = target.index_select(1, vis_idx)
        return torch.cat((target[:, :5], x_target, z_target, vis_target), dim=1)

    def _empty_loss(self, proposals: Tensor, focal_loss: FocalLoss, smooth_l1_loss: nn.SmoothL1Loss) -> dict[str, Tensor | None]:
        cls_target = proposals.new_zeros(len(proposals)).long()
        cls_pred = proposals[:, 5 + self.anchor_len * 3:]
        return {
            "cls_loss": focal_loss(cls_pred, cls_target).sum(),
            "reg_losses_x": smooth_l1_loss(cls_pred, cls_pred).sum() * 0,
            "reg_losses_z": smooth_l1_loss(cls_pred, cls_pred).sum() * 0,
            "reg_losses_vis": smooth_l1_loss(cls_pred, cls_pred).sum() * 0,
            "positives": proposals.new_tensor(0.0),
            "negatives": proposals.new_tensor(float(len(proposals))),
            "class_index_issues": proposals.new_tensor(0.0),
            "uncertainty_match": None,
        }

    def _match_and_collect(self, proposals: Tensor, anchors: Tensor, target: Tensor):
        with torch.no_grad():
            base = anchors if self.anchor_assign else proposals[:, :5 + self.anchor_len * 3]
            return self.assigner.match_proposals_with_targets(base, target)

    def _sanitize_cls_target(self, cls_target: Tensor, num_classes: int) -> tuple[Tensor, int]:
        if num_classes <= 0:
            raise ValueError(f"分类头维度非法: {num_classes}")
        if self.strict_class_check == "off":
            return cls_target.clamp(min=0, max=num_classes - 1), 0
        invalid = (cls_target < 0) | (cls_target >= num_classes)
        invalid_count = int(invalid.sum().item())
        if invalid_count <= 0:
            return cls_target, 0
        self.class_index_issue_total += invalid_count
        if self.strict_class_check == "raise":
            raise ValueError(f"检测到类别索引越界: count={invalid_count}, num_classes={num_classes}")
        if self.strict_class_check == "warn" and not self._warned_class_issue:
            warnings.warn("检测到类别索引越界，已执行安全收敛并累计计数。", RuntimeWarning)
            self._warned_class_issue = True
        return cls_target.clamp(min=0, max=num_classes - 1), invalid_count

    def _compute_sample_loss(
        self,
        proposals: Tensor,
        anchors: Tensor,
        target: Tensor,
        focal_loss: FocalLoss,
        smooth_l1_loss: nn.SmoothL1Loss,
    ) -> dict[str, Tensor | None]:
        target = target[target[:, 1] > 0]
        if len(target) == 0:
            return self._empty_loss(proposals, focal_loss, smooth_l1_loss)
        target = self._slice_target(target)
        pos_mask, neg_mask, pos_target_idx = self._match_and_collect(proposals, anchors, target)
        positives = proposals[pos_mask]
        negatives = proposals[neg_mask]
        num_pos = len(positives)
        if num_pos == 0:
            return self._empty_loss(proposals, focal_loss, smooth_l1_loss)
        all_proposals = torch.cat([positives, negatives], dim=0)
        cls_target = proposals.new_zeros(len(all_proposals)).long()
        cls_target[:num_pos] = target[pos_target_idx][:, 1]
        cls_pred = all_proposals[:, 5 + self.anchor_len * 3:]
        cls_target, issue_count = self._sanitize_cls_target(cls_target, cls_pred.shape[1])
        pos_target = target[pos_target_idx]
        x_pred = positives[:, 5:5 + self.anchor_len]
        z_pred = positives[:, 5 + self.anchor_len:5 + self.anchor_len * 2]
        vis_pred = positives[:, 5 + self.anchor_len * 2:5 + self.anchor_len * 3]
        x_target = pos_target[:, 5:5 + self.anchor_len]
        z_target = pos_target[:, 5 + self.anchor_len:5 + self.anchor_len * 2]
        vis_target = pos_target[:, 5 + self.anchor_len * 2:5 + self.anchor_len * 3]
        valid_points = vis_target.sum().clamp_min(1.0)
        reg_x = (smooth_l1_loss(x_pred, x_target) * vis_target).sum() / valid_points
        reg_z = (smooth_l1_loss(z_pred, z_target) * vis_target).sum() / valid_points
        reg_vis = smooth_l1_loss(vis_pred, vis_target).mean()
        cls_loss = focal_loss(cls_pred, cls_target).sum()
        if self.use_sigmoid:
            num_clses = cls_pred.shape[1]
            cls_loss = cls_loss / num_pos / num_clses
        else:
            cls_loss = cls_loss / num_pos
        return {
            "cls_loss": cls_loss,
            "reg_losses_x": reg_x,
            "reg_losses_z": reg_z,
            "reg_losses_vis": reg_vis,
            "positives": proposals.new_tensor(float(num_pos)),
            "negatives": proposals.new_tensor(float(len(negatives))),
            "class_index_issues": proposals.new_tensor(float(issue_count)),
            "uncertainty_match": (pos_mask.nonzero(as_tuple=False).squeeze(1), pos_target),
        }

    def forward(self, proposals_list: list[tuple[Tensor, Tensor]], targets: list[Tensor]) -> dict[str, Tensor | list[tuple[Tensor, Tensor] | None]]:
        focal_loss = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        smooth_l1_loss = nn.SmoothL1Loss(reduction="none")
        cls_losses = []
        reg_x_losses = []
        reg_z_losses = []
        reg_vis_losses = []
        positives = []
        negatives = []
        class_index_issues = []
        uncertainty_matches: list[tuple[Tensor, Tensor] | None] = []
        for (proposal, anchor), target in zip(proposals_list, targets):
            sample_loss = self._compute_sample_loss(proposal, anchor, target, focal_loss, smooth_l1_loss)
            cls_losses.append(sample_loss["cls_loss"])
            reg_x_losses.append(sample_loss["reg_losses_x"])
            reg_z_losses.append(sample_loss["reg_losses_z"])
            reg_vis_losses.append(sample_loss["reg_losses_vis"])
            positives.append(sample_loss["positives"])
            negatives.append(sample_loss["negatives"])
            class_index_issues.append(sample_loss["class_index_issues"])
            uncertainty_matches.append(sample_loss["uncertainty_match"])
        losses = {
            "cls_loss": torch.stack(cls_losses).mean() * self.loss_weights["cls_loss"],
            "reg_losses_x": torch.stack(reg_x_losses).mean() * self.loss_weights["reg_losses_x"],
            "reg_losses_z": torch.stack(reg_z_losses).mean() * self.loss_weights["reg_losses_z"],
            "reg_losses_vis": torch.stack(reg_vis_losses).mean() * self.loss_weights["reg_losses_vis"],
        }
        return {
            "losses": losses,
            "batch_positives": torch.stack(positives).mean(),
            "batch_negatives": torch.stack(negatives).mean(),
            "class_index_issues": torch.stack(class_index_issues).sum(),
            "uncertainty_matches": uncertainty_matches,
        }
