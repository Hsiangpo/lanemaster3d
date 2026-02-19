from __future__ import annotations

import numpy as np


def _sort_lane_by_y(lane: np.ndarray) -> np.ndarray:
    if lane.shape[0] < 2:
        return lane
    order = np.argsort(lane[:, 1])
    lane = lane[order]
    _, unique_idx = np.unique(lane[:, 1], return_index=True)
    return lane[np.sort(unique_idx)]


def _interp_with_extrapolation(src_y: np.ndarray, src_v: np.ndarray, dst_y: np.ndarray) -> np.ndarray:
    out = np.interp(dst_y, src_y, src_v)
    if src_y.shape[0] < 2:
        return out
    left_mask = dst_y < src_y[0]
    right_mask = dst_y > src_y[-1]
    if left_mask.any():
        slope = (src_v[1] - src_v[0]) / max(src_y[1] - src_y[0], 1e-6)
        out[left_mask] = src_v[0] + (dst_y[left_mask] - src_y[0]) * slope
    if right_mask.any():
        slope = (src_v[-1] - src_v[-2]) / max(src_y[-1] - src_y[-2], 1e-6)
        out[right_mask] = src_v[-1] + (dst_y[right_mask] - src_y[-1]) * slope
    return out


def prune_3d_lane_by_visibility(lane_3d: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    if lane_3d.shape[0] == 0:
        return lane_3d
    mask = visibility > 0
    if mask.shape[0] != lane_3d.shape[0]:
        return lane_3d
    return lane_3d[mask]


def prune_3d_lane_by_range(lane_3d: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    if lane_3d.shape[0] == 0:
        return lane_3d
    lane = lane_3d[lane_3d[:, 1] > 0]
    lane = lane[np.logical_and(lane[:, 0] >= x_min, lane[:, 0] <= x_max)]
    return lane


def resample_laneline_in_y(input_lane: np.ndarray, y_steps: np.ndarray, out_vis: bool = False):
    lane = np.asarray(input_lane, dtype=np.float32)
    if lane.ndim != 2 or lane.shape[1] != 3:
        raise ValueError("车道点必须是 [N,3] 形状")
    lane = _sort_lane_by_y(lane)
    if lane.shape[0] < 2:
        x_vals = np.zeros_like(y_steps, dtype=np.float32)
        z_vals = np.zeros_like(y_steps, dtype=np.float32)
        vis = np.zeros_like(y_steps, dtype=np.float32)
        return (x_vals, z_vals, vis) if out_vis else (x_vals, z_vals)
    x_vals = _interp_with_extrapolation(lane[:, 1], lane[:, 0], y_steps).astype(np.float32)
    z_vals = _interp_with_extrapolation(lane[:, 1], lane[:, 2], y_steps).astype(np.float32)
    y_min, y_max = lane[:, 1].min(), lane[:, 1].max()
    vis = np.logical_and(y_steps >= y_min, y_steps <= y_max).astype(np.float32)
    if out_vis:
        return x_vals, z_vals, vis
    return x_vals, z_vals
