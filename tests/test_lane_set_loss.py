from __future__ import annotations

import torch

from lanemaster3d.models.losses import LaneSetLoss


def _build_target(num_points: int, category: int) -> torch.Tensor:
    target = torch.zeros(1, 5 + num_points * 3, dtype=torch.float32)
    target[0, 1] = float(category)
    target[0, 4] = float(num_points)
    x = torch.linspace(-1.0, 1.0, steps=num_points)
    z = torch.zeros(num_points, dtype=torch.float32)
    vis = torch.ones(num_points, dtype=torch.float32)
    target[0, 5:5 + num_points] = x
    target[0, 5 + num_points:5 + num_points * 2] = z
    target[0, 5 + num_points * 2:5 + num_points * 3] = vis
    return target


def _build_proposals(num_points: int, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    proposals = torch.zeros(4, 5 + num_points * 3 + num_classes, dtype=torch.float32)
    anchors = torch.zeros(4, 5 + num_points * 3, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, steps=num_points)
    proposals[0, 5:5 + num_points] = x
    anchors[0, 5:5 + num_points] = x
    proposals[:, 4] = float(num_points)
    anchors[:, 4] = float(num_points)
    proposals[1:, 5:5 + num_points] = 10.0
    anchors[1:, 5:5 + num_points] = 10.0
    return proposals, anchors


def test_lane_set_loss_supports_category_21_when_num_classes_22() -> None:
    num_points = 6
    lane_loss = LaneSetLoss(
        anchor_len=num_points,
        gt_anchor_len=num_points,
        anchor_steps=list(range(1, num_points + 1)),
        assign_cfg={"type": "TopKLaneAssigner", "pos_k": 1, "neg_k": 2, "anchor_len": num_points},
    )
    proposals, anchors = _build_proposals(num_points=num_points, num_classes=22)
    target = _build_target(num_points=num_points, category=21)
    result = lane_loss([(proposals, anchors)], [target])
    cls_loss = result["losses"]["cls_loss"]
    assert torch.isfinite(cls_loss)


def test_lane_set_loss_clamps_category_index_when_head_is_21() -> None:
    num_points = 6
    lane_loss = LaneSetLoss(
        anchor_len=num_points,
        gt_anchor_len=num_points,
        anchor_steps=list(range(1, num_points + 1)),
        assign_cfg={"type": "TopKLaneAssigner", "pos_k": 1, "neg_k": 2, "anchor_len": num_points},
    )
    proposals, anchors = _build_proposals(num_points=num_points, num_classes=21)
    target = _build_target(num_points=num_points, category=21)
    result = lane_loss([(proposals, anchors)], [target])
    assert torch.isfinite(result["losses"]["cls_loss"])


def test_lane_set_loss_raise_on_invalid_category_when_strict() -> None:
    num_points = 6
    lane_loss = LaneSetLoss(
        anchor_len=num_points,
        gt_anchor_len=num_points,
        anchor_steps=list(range(1, num_points + 1)),
        assign_cfg={"type": "TopKLaneAssigner", "pos_k": 1, "neg_k": 2, "anchor_len": num_points},
        strict_class_check="raise",
    )
    proposals, anchors = _build_proposals(num_points=num_points, num_classes=21)
    target = _build_target(num_points=num_points, category=21)
    try:
        _ = lane_loss([(proposals, anchors)], [target])
        assert False, "strict 模式应抛出类别越界异常"
    except ValueError:
        assert True


def test_lane_set_loss_warn_mode_reports_issue_count() -> None:
    num_points = 6
    lane_loss = LaneSetLoss(
        anchor_len=num_points,
        gt_anchor_len=num_points,
        anchor_steps=list(range(1, num_points + 1)),
        assign_cfg={"type": "TopKLaneAssigner", "pos_k": 1, "neg_k": 2, "anchor_len": num_points},
        strict_class_check="warn",
    )
    proposals, anchors = _build_proposals(num_points=num_points, num_classes=21)
    target = _build_target(num_points=num_points, category=21)
    result = lane_loss([(proposals, anchors)], [target])
    assert int(result["class_index_issues"].item()) > 0


def test_lane_set_loss_returns_uncertainty_matches() -> None:
    num_points = 6
    lane_loss = LaneSetLoss(
        anchor_len=num_points,
        gt_anchor_len=num_points,
        anchor_steps=list(range(1, num_points + 1)),
        assign_cfg={"type": "TopKLaneAssigner", "pos_k": 1, "neg_k": 2, "anchor_len": num_points},
    )
    proposals, anchors = _build_proposals(num_points=num_points, num_classes=22)
    target = _build_target(num_points=num_points, category=1)
    result = lane_loss([(proposals, anchors)], [target])
    assert "uncertainty_matches" in result
    assert len(result["uncertainty_matches"]) == 1
    matched = result["uncertainty_matches"][0]
    assert matched is not None
    pos_index, pos_target = matched
    assert int(pos_index.numel()) > 0
    assert int(pos_target.shape[1]) == 5 + num_points * 3
