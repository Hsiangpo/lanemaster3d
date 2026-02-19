from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..data import OpenLaneDataset, openlane_collate
from ..metrics import build_openlane_prediction_line, evaluate_lane_batch, evaluate_openlane_official
from ..models import LaneMaster3DNet
from .common import build_project_matrix, to_device


@dataclass
class DistContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def _is_main(ctx: DistContext) -> bool:
    return ctx.rank == 0


def _setup_distributed(config: dict, launcher: str, device: str) -> DistContext:
    if launcher != "ddp":
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        return DistContext(False, 0, 1, 0, dev)
    backend = config.get("distributed", {}).get("backend", "nccl" if torch.cuda.is_available() else "gloo")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
    else:
        dev = torch.device("cpu")
    return DistContext(True, rank, world_size, local_rank, dev)


def _cleanup_distributed(ctx: DistContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _build_loader(config: dict, ctx: DistContext) -> DataLoader:
    data_cfg = config["data"]
    runtime = config["runtime"]
    batch_size = int(
        runtime.get(
            "official_eval_batch_size_per_gpu",
            runtime.get("val_batch_size_per_gpu", runtime.get("batch_size_per_gpu", runtime.get("batch_size", 4))),
        )
    )
    workers = int(runtime.get("official_eval_num_workers", runtime.get("val_num_workers", runtime.get("num_workers", 0))))
    dataset = OpenLaneDataset(
        data_root=data_cfg["data_root"],
        list_path=data_cfg["val_list"],
        image_size=tuple(data_cfg["image_size"]),
        max_lanes=data_cfg["max_lanes"],
        num_points=data_cfg["num_points"],
        camera_param_policy=data_cfg.get("camera_param_policy", "strict"),
        category_policy=data_cfg.get("category_policy", "preserve_21"),
        preindex_cache=bool(data_cfg.get("preindex_cache", True)),
    )
    sampler = DistributedSampler(dataset, shuffle=False) if ctx.enabled else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(runtime.get("pin_memory", True)),
        collate_fn=openlane_collate,
    )


def _build_model(config: dict, device: torch.device) -> LaneMaster3DNet:
    model_cfg = config["model"]
    model = LaneMaster3DNet(
        hidden_dim=model_cfg["hidden_dim"],
        query_count=model_cfg["query_count"],
        num_points=model_cfg["num_points"],
        num_category=int(model_cfg.get("num_category", 21)),
        use_gca=bool(config["innovation"]["gca"]["enabled"]),
        backbone_name=model_cfg.get("backbone_name", "resnet101"),
        dynamic_anchor_enabled=bool(config["innovation"]["dagp"]["enabled"]),
        dynamic_anchor_delta_scale=float(config["innovation"]["dagp"].get("delta_scale", 0.25)),
        y_steps=model_cfg.get("y_steps"),
        anchor_cfg=model_cfg.get("anchor_cfg"),
        iter_reg=int(model_cfg.get("iter_reg", 2)),
        anchor_feat_channels=int(model_cfg.get("anchor_feat_channels", 64)),
        feature_level=int(model_cfg.get("feature_level", 2)),
    )
    return model.to(device)


def _load_gt_from_meta(meta: dict[str, str], cache: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
    file_path = meta.get("file_path", meta.get("sample_id", ""))
    ann_path = meta.get("ann_path", "")
    if file_path in cache:
        return file_path, cache[file_path]
    if not ann_path:
        return file_path, None
    path = Path(ann_path)
    if not path.exists():
        return file_path, None
    gt = json.loads(path.read_text(encoding="utf-8"))
    gt["file_path"] = file_path
    cache[file_path] = gt
    return file_path, gt


def _build_prediction_lines(output: dict[str, torch.Tensor], metas: list[dict[str, str]], score_thresh: float):
    predictions = []
    scores = output["pred_scores"].detach().cpu().numpy()
    points = output["pred_points"].detach().cpu().numpy()
    vis = output["pred_vis"].detach().cpu().numpy()
    logits = output["pred_logits"].detach().cpu().numpy()
    for idx, meta in enumerate(metas):
        keep_inds = np.where(scores[idx] >= score_thresh)[0].tolist()
        lane_points = [points[idx, lane_idx] for lane_idx in keep_inds]
        lane_scores = [float(scores[idx, lane_idx]) for lane_idx in keep_inds]
        lane_vis = [vis[idx, lane_idx] for lane_idx in keep_inds]
        lane_cates = []
        for lane_idx in keep_inds:
            row = logits[idx, lane_idx]
            lane_cates.append(int(np.argmax(row[1:]) + 1) if row.shape[0] > 1 else 1)
        predictions.append(
            build_openlane_prediction_line(
                file_path=meta.get("file_path", meta.get("sample_id", "")),
                lane_points=lane_points,
                lane_scores=lane_scores,
                lane_categories=lane_cates,
                lane_vis=lane_vis,
            )
        )
    return predictions


def _average_quick_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "x_error": 0.0, "z_error": 0.0}
    keys = metrics[0].keys()
    return {key: float(np.mean([m.get(key, 0.0) for m in metrics])) for key in keys}


def _reduce_quick_metrics(quick: dict[str, float], sample_count: int, ctx: DistContext) -> dict[str, float]:
    if not ctx.enabled:
        return quick
    keys = sorted(quick.keys())
    values = [quick[key] * sample_count for key in keys]
    tensor = torch.tensor(values + [float(sample_count)], dtype=torch.float32, device=ctx.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    denom = max(float(tensor[-1].item()), 1.0)
    return {key: float(tensor[idx].item() / denom) for idx, key in enumerate(keys)}


def _gather_eval_objects(
    predictions: list[dict[str, Any]],
    gt_map: dict[str, dict[str, Any]],
    ctx: DistContext,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not ctx.enabled:
        return predictions, gt_map
    pred_chunks: list[list[dict[str, Any]] | None] = [None for _ in range(ctx.world_size)]
    gt_chunks: list[dict[str, dict[str, Any]] | None] = [None for _ in range(ctx.world_size)]
    dist.all_gather_object(pred_chunks, predictions)
    dist.all_gather_object(gt_chunks, gt_map)
    if not _is_main(ctx):
        return [], {}
    merged_predictions: list[dict[str, Any]] = []
    merged_gt_map: dict[str, dict[str, Any]] = {}
    for chunk in pred_chunks:
        if isinstance(chunk, list):
            merged_predictions.extend(chunk)
    for chunk in gt_chunks:
        if isinstance(chunk, dict):
            merged_gt_map.update(chunk)
    return merged_predictions, merged_gt_map


def evaluate_model(
    config: dict,
    checkpoint: str,
    output_dir: str,
    device: str = "cuda",
    launcher: str = "none",
) -> dict[str, float]:
    ctx = _setup_distributed(config, launcher=launcher, device=device)
    try:
        loader = _build_loader(config, ctx)
        model = _build_model(config, ctx.device)
        ckpt = torch.load(checkpoint, map_location=ctx.device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        score_thresh = float(config["runtime"].get("score_threshold", 0.5))
        quick_metrics: list[dict[str, float]] = []
        predictions: list[dict[str, Any]] = []
        gt_map: dict[str, dict[str, Any]] = {}
        with torch.no_grad():
            for raw_batch in loader:
                metas = raw_batch["meta"]
                batch = to_device(raw_batch, ctx.device)
                output = model(batch["image"], project_matrix=build_project_matrix(batch))
                quick_metrics.append(
                    evaluate_lane_batch(
                        output["pred_points"],
                        output["pred_scores"],
                        batch["gt_points"],
                        batch["lane_valid"],
                    )
                )
                predictions.extend(_build_prediction_lines(output, metas, score_thresh))
                for meta in metas:
                    file_path, gt = _load_gt_from_meta(meta, gt_map)
                    if gt is not None:
                        gt_map[file_path] = gt
        quick_local = _average_quick_metrics(quick_metrics)
        quick = _reduce_quick_metrics(quick_local, len(quick_metrics), ctx)
        predictions, gt_map = _gather_eval_objects(predictions, gt_map, ctx)
        if not _is_main(ctx):
            return {}
        use_official = bool(config["runtime"].get("official_eval", True))
        if use_official and gt_map:
            official = evaluate_openlane_official(predictions, gt_map, prob_th=score_thresh)
            result = {**official, "quick_f1": quick["f1"]}
        else:
            result = quick
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (out_dir / "predictions.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in predictions),
            encoding="utf-8",
        )
        return result
    finally:
        _cleanup_distributed(ctx)
