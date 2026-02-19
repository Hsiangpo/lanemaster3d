from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import prune_3d_lane_by_range, prune_3d_lane_by_visibility, resample_laneline_in_y
from .min_cost_flow import solve_min_cost_flow


def _to_lane_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        return np.zeros((0, 3), dtype=np.float32)
    if arr.shape[1] == 3:
        return arr
    if arr.shape[0] == 3:
        return arr.T
    return np.zeros((0, 3), dtype=np.float32)


def _safe_mean(values: np.ndarray) -> float:
    valid = values[values > -1 + 1e-6]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))


@dataclass
class EvalPack:
    lanes: list[np.ndarray]
    categories: list[int]
    visibility: np.ndarray


class OpenLaneOfficialEvaluator:
    """OpenLane官方协议评测器。"""

    def __init__(self) -> None:
        self.top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]], dtype=np.float32)
        self.x_min = float(self.top_view_region[0, 0])
        self.x_max = float(self.top_view_region[1, 0])
        self.y_min = float(self.top_view_region[2, 1])
        self.y_max = float(self.top_view_region[0, 1])
        self.y_samples = np.linspace(self.y_min, self.y_max, num=100, endpoint=False).astype(np.float32)
        self.dist_th = 1.5
        self.ratio_th = 0.75
        self.close_range = 40.0

    def _overlap_sampling_range(self, lane: np.ndarray) -> bool:
        if lane.shape[0] < 2:
            return False
        y_vals = lane[:, 1]
        return bool(y_vals.min() < self.y_samples[-1] and y_vals.max() > self.y_samples[0])

    def _prepare_group(
        self,
        lanes: list[np.ndarray],
        categories: list[int],
        visibilities: list[np.ndarray] | None = None,
    ) -> tuple[list[np.ndarray], list[int]]:
        out_lanes: list[np.ndarray] = []
        out_cates: list[int] = []
        for idx, lane in enumerate(lanes):
            work = lane
            if visibilities is not None and idx < len(visibilities):
                work = prune_3d_lane_by_visibility(work, visibilities[idx])
            if not self._overlap_sampling_range(work):
                continue
            work = prune_3d_lane_by_range(work, self.x_min, self.x_max)
            if work.shape[0] <= 1:
                continue
            out_lanes.append(work)
            out_cates.append(categories[idx] if idx < len(categories) else 1)
        return out_lanes, out_cates

    def _resample_group(self, lanes: list[np.ndarray], categories: list[int]) -> EvalPack:
        vis_rows: list[np.ndarray] = []
        lane_rows: list[np.ndarray] = []
        out_cates: list[int] = []
        for idx, lane in enumerate(lanes):
            x_vals, z_vals, vis = resample_laneline_in_y(lane, self.y_samples, out_vis=True)
            y_min, y_max = lane[:, 1].min(), lane[:, 1].max()
            in_range = np.logical_and(x_vals >= self.x_min, x_vals <= self.x_max)
            in_y = np.logical_and(self.y_samples >= y_min, self.y_samples <= y_max)
            lane_vis = np.logical_and(in_range, in_y)
            lane_vis = np.logical_and(lane_vis, vis > 0.5).astype(np.float32)
            if lane_vis.sum() <= 1:
                continue
            lane_rows.append(np.stack([x_vals, z_vals], axis=-1))
            vis_rows.append(lane_vis)
            out_cates.append(categories[idx] if idx < len(categories) else 1)
        vis_mat = np.stack(vis_rows, axis=0) if vis_rows else np.zeros((0, 100), dtype=np.float32)
        return EvalPack(lanes=lane_rows, categories=out_cates, visibility=vis_mat)

    def _pair_distance(
        self,
        gt_lane: np.ndarray,
        pred_lane: np.ndarray,
        gt_vis: np.ndarray,
        pred_vis: np.ndarray,
        close_idx: int,
    ) -> tuple[int, float, float, float, float, float]:
        x_dist = np.abs(gt_lane[:, 0] - pred_lane[:, 0])
        z_dist = np.abs(gt_lane[:, 1] - pred_lane[:, 1])
        both_visible = np.logical_and(gt_vis >= 0.5, pred_vis >= 0.5)
        both_invisible = np.logical_and(gt_vis < 0.5, pred_vis < 0.5)
        other = np.logical_not(np.logical_or(both_visible, both_invisible))
        euc = np.sqrt(x_dist**2 + z_dist**2)
        euc[both_invisible] = 0.0
        euc[other] = self.dist_th
        num_match = int(np.sum(euc < self.dist_th) - np.sum(both_invisible))
        cost = float(np.sum(euc))
        close = both_visible[:close_idx]
        far = both_visible[close_idx:]
        x_close = float(np.sum(x_dist[:close_idx] * close) / max(np.sum(close), 1)) if np.sum(close) > 0 else -1.0
        z_close = float(np.sum(z_dist[:close_idx] * close) / max(np.sum(close), 1)) if np.sum(close) > 0 else -1.0
        x_far = float(np.sum(x_dist[close_idx:] * far) / max(np.sum(far), 1)) if np.sum(far) > 0 else -1.0
        z_far = float(np.sum(z_dist[close_idx:] * far) / max(np.sum(far), 1)) if np.sum(far) > 0 else -1.0
        return num_match, cost, x_close, x_far, z_close, z_far

    def _build_match_matrix(self, gt: EvalPack, pred: EvalPack):
        gt_n = len(gt.lanes)
        pred_n = len(pred.lanes)
        adj = np.zeros((gt_n, pred_n), dtype=np.int32)
        cost = np.full((gt_n, pred_n), 1000, dtype=np.int32)
        num_match = np.zeros((gt_n, pred_n), dtype=np.float32)
        x_close = np.full((gt_n, pred_n), 1000.0, dtype=np.float32)
        x_far = np.full((gt_n, pred_n), 1000.0, dtype=np.float32)
        z_close = np.full((gt_n, pred_n), 1000.0, dtype=np.float32)
        z_far = np.full((gt_n, pred_n), 1000.0, dtype=np.float32)
        close_idx = int(np.where(self.y_samples > self.close_range)[0][0])
        for i in range(gt_n):
            for j in range(pred_n):
                matched, pair_cost, xc, xf, zc, zf = self._pair_distance(
                    gt.lanes[i], pred.lanes[j], gt.visibility[i], pred.visibility[j], close_idx
                )
                adj[i, j] = 1
                cost[i, j] = int(pair_cost if pair_cost >= 1 else 1)
                num_match[i, j] = matched
                x_close[i, j], x_far[i, j] = xc, xf
                z_close[i, j], z_far[i, j] = zc, zf
        return adj, cost, num_match, x_close, x_far, z_close, z_far

    def bench(
        self,
        pred_lanes: list[np.ndarray],
        pred_category: list[int],
        gt_lanes: list[np.ndarray],
        gt_visibility: list[np.ndarray],
        gt_category: list[int],
    ):
        gt_lanes, gt_category = self._prepare_group(gt_lanes, gt_category, gt_visibility)
        pred_lanes, pred_category = self._prepare_group(pred_lanes, pred_category, None)
        gt_pack = self._resample_group(gt_lanes, gt_category)
        pred_pack = self._resample_group(pred_lanes, pred_category)
        cnt_gt, cnt_pred = len(gt_pack.lanes), len(pred_pack.lanes)
        if cnt_gt == 0 or cnt_pred == 0:
            return 0.0, 0.0, 0.0, cnt_gt, cnt_pred, 0.0, [], [], [], []
        matrices = self._build_match_matrix(gt_pack, pred_pack)
        adj, cost, num_match, x_close, x_far, z_close, z_far = matrices
        results = np.asarray(solve_min_cost_flow(adj, cost), dtype=np.float32)
        return self._collect_stats(results, gt_pack, pred_pack, num_match, x_close, x_far, z_close, z_far)

    def _collect_stats(self, results: np.ndarray, gt: EvalPack, pred: EvalPack, num_match, x_close, x_far, z_close, z_far):
        r_lane, p_lane, c_lane = 0.0, 0.0, 0.0
        x_close_all, x_far_all, z_close_all, z_far_all = [], [], [], []
        match_num = 0.0
        if results.size == 0:
            return r_lane, p_lane, c_lane, len(gt.lanes), len(pred.lanes), match_num, x_close_all, x_far_all, z_close_all, z_far_all
        for row in results:
            gt_i, pred_i, pair_cost = int(row[0]), int(row[1]), float(row[2])
            if pair_cost >= self.dist_th * self.y_samples.shape[0]:
                continue
            match_num += 1
            gt_ratio = num_match[gt_i, pred_i] / max(np.sum(gt.visibility[gt_i]), 1)
            pred_ratio = num_match[gt_i, pred_i] / max(np.sum(pred.visibility[pred_i]), 1)
            if gt_ratio >= self.ratio_th:
                r_lane += 1
            if pred_ratio >= self.ratio_th:
                p_lane += 1
            gt_c = gt.categories[gt_i] if gt_i < len(gt.categories) else 1
            pred_c = pred.categories[pred_i] if pred_i < len(pred.categories) else 1
            if pred_c == gt_c or (pred_c == 20 and gt_c == 21):
                c_lane += 1
            x_close_all.append(x_close[gt_i, pred_i])
            x_far_all.append(x_far[gt_i, pred_i])
            z_close_all.append(z_close[gt_i, pred_i])
            z_far_all.append(z_far[gt_i, pred_i])
        return r_lane, p_lane, c_lane, len(gt.lanes), len(pred.lanes), match_num, x_close_all, x_far_all, z_close_all, z_far_all

    def _transform_gt_lane(self, lane_xyz: np.ndarray, cam_extrinsics: np.ndarray) -> np.ndarray:
        if lane_xyz.shape[0] == 0:
            return lane_xyz
        lane = np.vstack((lane_xyz.T, np.ones((1, lane_xyz.shape[0]), dtype=np.float32)))
        cam_rep = np.linalg.inv(
            np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        )
        lane = cam_extrinsics @ (cam_rep @ lane)
        return lane[0:3, :].T

    def _prepare_gt_sample(self, gt: dict[str, Any]):
        cam_extrinsics = None
        extr = gt.get("extrinsic")
        if isinstance(extr, list):
            raw_extr = np.asarray(extr, dtype=np.float32)
            if raw_extr.shape == (4, 4):
                cam_extrinsics = raw_extr.copy()
                r_vg = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float32)
                r_gc = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
                cam_extrinsics[:3, :3] = np.linalg.inv(r_vg) @ cam_extrinsics[:3, :3] @ r_vg @ r_gc
                cam_extrinsics[0:2, 3] = 0.0
        gt_lanes, gt_vis, gt_cates = [], [], []
        for lane_obj in gt.get("lane_lines", []):
            lane = _to_lane_array(lane_obj.get("xyz", []))
            vis = np.asarray(lane_obj.get("visibility", np.ones((lane.shape[0],))), dtype=np.float32)
            if vis.shape[0] != lane.shape[0]:
                vis = np.ones((lane.shape[0],), dtype=np.float32)
            gt_lanes.append(self._transform_gt_lane(lane, cam_extrinsics) if cam_extrinsics is not None else lane)
            gt_vis.append(vis)
            gt_cates.append(int(lane_obj.get("category", 1)))
        return gt_lanes, gt_vis, gt_cates

    def bench_one_submit(self, json_pred: list[dict[str, Any]], gts: dict[str, dict[str, Any]], prob_th: float = 0.5):
        stats = []
        x_close_all: list[float] = []
        x_far_all: list[float] = []
        z_close_all: list[float] = []
        z_far_all: list[float] = []
        for pred in json_pred:
            raw_file = pred.get("file_path")
            if not raw_file or raw_file not in gts:
                continue
            pred_lanes, pred_cates = [], []
            for lane_obj in pred.get("lane_lines", []):
                prob = float(lane_obj.get("laneLines_prob", 0.0))
                if prob <= prob_th:
                    continue
                pred_lanes.append(_to_lane_array(lane_obj.get("xyz", [])))
                pred_cates.append(int(lane_obj.get("category", 1)))
            gt_lanes, gt_vis, gt_cates = self._prepare_gt_sample(gts[raw_file])
            bench = self.bench(pred_lanes, pred_cates, gt_lanes, gt_vis, gt_cates)
            r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, x_close, x_far, z_close, z_far = bench
            stats.append(np.array([r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num], dtype=np.float32))
            x_close_all.extend(x_close)
            x_far_all.extend(x_far)
            z_close_all.extend(z_close)
            z_far_all.extend(z_far)
        return self._finalize(stats, x_close_all, x_far_all, z_close_all, z_far_all)

    def _finalize(self, stats: list[np.ndarray], x_close: list[float], x_far: list[float], z_close: list[float], z_far: list[float]):
        if not stats:
            return [0.0] * 14
        arr = np.stack(stats, axis=0)
        recall = float(arr[:, 0].sum() / max(arr[:, 3].sum(), 1e-6))
        precision = float(arr[:, 1].sum() / max(arr[:, 4].sum(), 1e-6))
        cate_acc = float(arr[:, 2].sum() / max(arr[:, 5].sum(), 1e-6))
        f_score = float(2 * recall * precision / max(recall + precision, 1e-6))
        out = [f_score, recall, precision, cate_acc]
        out.append(_safe_mean(np.asarray(x_close, dtype=np.float32)))
        out.append(_safe_mean(np.asarray(x_far, dtype=np.float32)))
        out.append(_safe_mean(np.asarray(z_close, dtype=np.float32)))
        out.append(_safe_mean(np.asarray(z_far, dtype=np.float32)))
        out.extend(
            [
                float(arr[:, 0].sum()),
                float(arr[:, 1].sum()),
                float(arr[:, 2].sum()),
                float(arr[:, 3].sum()),
                float(arr[:, 4].sum()),
                float(arr[:, 5].sum()),
            ]
        )
        return out
