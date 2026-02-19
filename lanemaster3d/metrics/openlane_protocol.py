from __future__ import annotations

from typing import Any

import numpy as np

from .openlane_official import OpenLaneOfficialEvaluator


def _to_xyz_list(points: np.ndarray) -> list[list[float]]:
    lane = np.asarray(points, dtype=np.float32)
    if lane.ndim != 2 or lane.shape[1] != 3:
        return []
    return lane.tolist()


def build_openlane_prediction_line(
    file_path: str,
    lane_points: list[np.ndarray],
    lane_scores: list[float],
    lane_categories: list[int] | None = None,
    lane_vis: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    categories = lane_categories or [1 for _ in lane_points]
    vis_list = lane_vis or [None for _ in lane_points]
    lane_lines = []
    for idx, lane in enumerate(lane_points):
        xyz = np.asarray(lane, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            continue
        vis = vis_list[idx] if idx < len(vis_list) else None
        if isinstance(vis, np.ndarray) and vis.shape[0] == xyz.shape[0]:
            xyz = xyz[vis > 0.5]
        if xyz.shape[0] < 2:
            continue
        lane_lines.append(
            {
                "xyz": _to_xyz_list(xyz),
                "category": int(categories[idx] if idx < len(categories) else 1),
                "laneLines_prob": float(lane_scores[idx] if idx < len(lane_scores) else 0.0),
            }
        )
    return {"file_path": file_path, "lane_lines": lane_lines}


def evaluate_openlane_official(
    predictions: list[dict[str, Any]],
    gts: dict[str, dict[str, Any]],
    prob_th: float = 0.5,
) -> dict[str, float]:
    evaluator = OpenLaneOfficialEvaluator()
    stats = evaluator.bench_one_submit(predictions, gts, prob_th=prob_th)
    return {
        "F_score": float(stats[0]),
        "recall": float(stats[1]),
        "precision": float(stats[2]),
        "cate_acc": float(stats[3]),
        "x_error_close": float(stats[4]),
        "x_error_far": float(stats[5]),
        "z_error_close": float(stats[6]),
        "z_error_far": float(stats[7]),
    }
