from __future__ import annotations

import torch
from torch import Tensor


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _resolve_src_hw(intrinsic: Tensor, src_img_hw: Tensor | None, image_hw: tuple[int, int]) -> tuple[Tensor, Tensor]:
    image_h, image_w = image_hw
    image_h_tensor = intrinsic.new_full((intrinsic.shape[0],), float(max(image_h, 1)))
    image_w_tensor = intrinsic.new_full((intrinsic.shape[0],), float(max(image_w, 1)))
    intrinsic_w = (intrinsic[:, 0, 2] * 2.0).clamp(min=1.0)
    intrinsic_h = (intrinsic[:, 1, 2] * 2.0).clamp(min=1.0)
    default_h = torch.where(image_h_tensor > 0, image_h_tensor, intrinsic_h)
    default_w = torch.where(image_w_tensor > 0, image_w_tensor, intrinsic_w)
    if not isinstance(src_img_hw, Tensor):
        return default_h, default_w
    if src_img_hw.ndim != 2 or src_img_hw.shape[-1] != 2:
        return default_h, default_w
    src_h = src_img_hw[:, 0].to(intrinsic.device, dtype=intrinsic.dtype)
    src_w = src_img_hw[:, 1].to(intrinsic.device, dtype=intrinsic.dtype)
    src_h = torch.where(src_h > 0, src_h, default_h)
    src_w = torch.where(src_w > 0, src_w, default_w)
    return src_h, src_w


def _scale_intrinsic_to_input(intrinsic: Tensor, image: Tensor, src_img_hw: Tensor | None = None) -> Tensor:
    _, _, img_h, img_w = image.shape
    scaled = intrinsic.clone()
    src_h, src_w = _resolve_src_hw(intrinsic, src_img_hw, (img_h, img_w))
    scale_x = float(img_w) / src_w
    scale_y = float(img_h) / src_h
    scaled[:, 0, :] = scaled[:, 0, :] * scale_x[:, None]
    scaled[:, 1, :] = scaled[:, 1, :] * scale_y[:, None]
    return scaled


def build_project_matrix(batch: dict) -> Tensor | None:
    image = batch.get("image")
    extrinsic = batch.get("cam_extrinsic")
    intrinsic = batch.get("cam_intrinsic")
    src_img_hw = batch.get("src_img_hw")
    if not isinstance(image, Tensor):
        return None
    if not isinstance(extrinsic, Tensor):
        return None
    if not isinstance(intrinsic, Tensor):
        return None
    if extrinsic.ndim != 3 or intrinsic.ndim != 3:
        return None
    if extrinsic.shape[-2:] != (4, 4) or intrinsic.shape[-2:] != (3, 3):
        return None
    intrinsic_scaled = _scale_intrinsic_to_input(intrinsic, image, src_img_hw if isinstance(src_img_hw, Tensor) else None)
    extrinsic_34 = extrinsic[:, :3, :4]
    return torch.bmm(intrinsic_scaled, extrinsic_34)


def _build_from_points(pred_points: Tensor, pred_scores: Tensor, pred_logits: Tensor | None = None) -> list[tuple[Tensor, Tensor]]:
    batch_size, query_count, num_points, _ = pred_points.shape
    num_category = int(pred_logits.shape[-1]) if isinstance(pred_logits, Tensor) else 2
    proposal_list: list[tuple[Tensor, Tensor]] = []
    cls_logits = pred_logits if isinstance(pred_logits, Tensor) else torch.stack(
        [torch.log(1.0 - pred_scores.clamp(1e-4, 1 - 1e-4)), torch.log(pred_scores.clamp(1e-4, 1 - 1e-4))], dim=-1
    )
    for batch_idx in range(batch_size):
        proposal = pred_points.new_zeros(query_count, 5 + num_points * 3 + num_category)
        anchor = pred_points.new_zeros(query_count, 5 + num_points * 3)
        proposal[:, 1] = pred_scores[batch_idx]
        proposal[:, 4] = float(num_points)
        anchor[:, 4] = float(num_points)
        proposal[:, 5:5 + num_points] = pred_points[batch_idx, :, :, 0]
        proposal[:, 5 + num_points:5 + num_points * 2] = pred_points[batch_idx, :, :, 2]
        proposal[:, 5 + num_points * 2:5 + num_points * 3] = 1.0
        proposal[:, 5 + num_points * 3:5 + num_points * 3 + num_category] = cls_logits[batch_idx]
        anchor[:, 5:5 + num_points] = pred_points[batch_idx, :, :, 0].detach()
        anchor[:, 5 + num_points:5 + num_points * 2] = pred_points[batch_idx, :, :, 2].detach()
        anchor[:, 5 + num_points * 2:5 + num_points * 3] = 1.0
        proposal_list.append((proposal, anchor))
    return proposal_list


def build_proposals(output: dict[str, Tensor]) -> list[tuple[Tensor, Tensor]]:
    proposals = output.get("reg_proposals")
    anchors = output.get("anchors")
    if isinstance(proposals, Tensor) and isinstance(anchors, Tensor):
        return [(proposals[i], anchors[i]) for i in range(proposals.shape[0])]
    pred_points = output.get("pred_points")
    pred_scores = output.get("pred_scores")
    pred_logits = output.get("pred_logits")
    if not isinstance(pred_points, Tensor) or not isinstance(pred_scores, Tensor):
        raise ValueError("模型输出缺少 reg_proposals/anchors 或 pred_points/pred_scores")
    return _build_from_points(pred_points, pred_scores, pred_logits if isinstance(pred_logits, Tensor) else None)


def build_targets(gt_points: Tensor, gt_vis: Tensor, lane_valid: Tensor, gt_category: Tensor | None = None) -> list[Tensor]:
    batch_size, max_lanes, num_points, _ = gt_points.shape
    targets = []
    for batch_idx in range(batch_size):
        target = gt_points.new_zeros(max_lanes, 5 + num_points * 3)
        if gt_category is None:
            target[:, 1] = lane_valid[batch_idx]
        else:
            target[:, 1] = gt_category[batch_idx] * lane_valid[batch_idx]
        target[:, 4] = gt_vis[batch_idx].sum(dim=1)
        target[:, 5:5 + num_points] = gt_points[batch_idx, :, :, 0]
        target[:, 5 + num_points:5 + num_points * 2] = gt_points[batch_idx, :, :, 2]
        target[:, 5 + num_points * 2:5 + num_points * 3] = gt_vis[batch_idx]
        targets.append(target)
    return targets
