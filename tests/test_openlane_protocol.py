from __future__ import annotations

import numpy as np

from lanemaster3d.metrics.openlane_protocol import (
    build_openlane_prediction_line,
    evaluate_openlane_official,
)


def _lane_points() -> np.ndarray:
    y = np.array([5.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    x = np.array([0.0, 0.1, 0.2, 0.2, 0.3], dtype=np.float32)
    z = np.array([0.0, 0.0, 0.1, 0.1, 0.1], dtype=np.float32)
    return np.stack([x, y, z], axis=-1)


def test_openlane_prediction_format() -> None:
    lane = _lane_points()
    pred = build_openlane_prediction_line(
        file_path="segment-001/sample-001.jpg",
        lane_points=[lane],
        lane_scores=[0.95],
        lane_categories=[1],
    )
    assert pred["file_path"] == "segment-001/sample-001.jpg"
    assert len(pred["lane_lines"]) == 1
    assert "xyz" in pred["lane_lines"][0]
    assert "laneLines_prob" in pred["lane_lines"][0]


def test_openlane_official_eval_perfect_match() -> None:
    lane = _lane_points()
    pred = build_openlane_prediction_line(
        file_path="segment-001/sample-001.jpg",
        lane_points=[lane],
        lane_scores=[0.99],
        lane_categories=[1],
    )
    gt = {
        "file_path": "segment-001/sample-001.jpg",
        "lane_lines": [
            {
                "xyz": lane.T.tolist(),
                "visibility": [1, 1, 1, 1, 1],
                "category": 1,
            }
        ],
    }
    metrics = evaluate_openlane_official([pred], {"segment-001/sample-001.jpg": gt})
    assert metrics["F_score"] > 0.95
