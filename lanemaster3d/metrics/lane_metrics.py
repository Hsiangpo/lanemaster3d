from __future__ import annotations

import torch
from torch import Tensor


def _match_lanes(pred_points: Tensor, gt_points: Tensor, threshold: float) -> tuple[int, int, int, Tensor]:
    if pred_points.numel() == 0 and gt_points.numel() == 0:
        return 0, 0, 0, torch.empty(0, 2, dtype=torch.long, device=pred_points.device)
    if pred_points.numel() == 0:
        return 0, 0, gt_points.shape[0], torch.empty(0, 2, dtype=torch.long, device=gt_points.device)
    if gt_points.numel() == 0:
        return 0, pred_points.shape[0], 0, torch.empty(0, 2, dtype=torch.long, device=pred_points.device)
    distances = (pred_points[:, None] - gt_points[None]).abs().mean(dim=(2, 3))
    matched = []
    used_pred = set()
    used_gt = set()
    flat_inds = distances.flatten().argsort()
    for ind in flat_inds.tolist():
        pred_idx = ind // distances.shape[1]
        gt_idx = ind % distances.shape[1]
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        if distances[pred_idx, gt_idx] > threshold:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matched.append((pred_idx, gt_idx))
    tp = len(matched)
    fp = pred_points.shape[0] - tp
    fn = gt_points.shape[0] - tp
    if tp == 0:
        pairs = torch.empty(0, 2, dtype=torch.long, device=pred_points.device)
    else:
        pairs = torch.tensor(matched, dtype=torch.long, device=pred_points.device)
    return tp, fp, fn, pairs


def evaluate_lane_batch(
    pred_points: Tensor,
    pred_scores: Tensor,
    gt_points: Tensor,
    lane_valid: Tensor,
    score_thresh: float = 0.5,
    match_thresh: float = 1.5,
) -> dict[str, float]:
    totals = {"tp": 0, "fp": 0, "fn": 0}
    x_err = []
    z_err = []
    for batch_idx in range(pred_points.shape[0]):
        keep = pred_scores[batch_idx] >= score_thresh
        preds = pred_points[batch_idx][keep]
        gt_keep = lane_valid[batch_idx] > 0.5
        gts = gt_points[batch_idx][gt_keep]
        tp, fp, fn, pairs = _match_lanes(preds, gts, match_thresh)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        for pred_idx, gt_idx in pairs.tolist():
            x_err.append((preds[pred_idx, :, 0] - gts[gt_idx, :, 0]).abs().mean().item())
            z_err.append((preds[pred_idx, :, 2] - gts[gt_idx, :, 2]).abs().mean().item())
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "f1": float(f1 * 100.0),
        "precision": float(precision * 100.0),
        "recall": float(recall * 100.0),
        "x_error": float(sum(x_err) / max(len(x_err), 1)),
        "z_error": float(sum(z_err) / max(len(z_err), 1)),
    }

