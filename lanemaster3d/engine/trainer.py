from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..data import OpenLaneDataset, openlane_collate
from ..metrics import build_openlane_prediction_line, evaluate_lane_batch, evaluate_openlane_official
from ..models import LaneMaster3DNet
from ..models.losses import HeteroscedasticLaneUncertaintyLoss, GeometryConsistencySelfSupervisionLoss, LaneSetLoss
from .common import build_project_matrix, build_proposals, build_targets, to_device
from .train_utils import average_metric_list as _average_metric_list, should_run_official_eval as _should_run_official_eval

@dataclass
class DistContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

def _is_main(ctx: DistContext) -> bool:
    return ctx.rank == 0

def _runtime(config: dict) -> dict:
    runtime = dict(config["runtime"])
    runtime.setdefault("amp", True)
    runtime.setdefault("accum_steps", 1)
    runtime.setdefault("log_interval", 20)
    runtime.setdefault("prefetch_factor", 4)
    runtime.setdefault("pin_memory", True)
    runtime.setdefault("persistent_workers", True)
    runtime.setdefault("amp_dtype", "float16")
    runtime.setdefault("tf32", True)
    runtime.setdefault("cudnn_benchmark", True)
    runtime.setdefault("channels_last", True)
    runtime.setdefault("ddp_static_graph", True)
    runtime.setdefault("compile_model", False)
    runtime.setdefault("compile_backend", "inductor")
    runtime.setdefault("compile_mode", "reduce-overhead")
    runtime.setdefault("compile_dynamic", False)
    runtime.setdefault("fused_adamw", True)
    runtime.setdefault("batch_size_per_gpu", runtime.get("batch_size", 4))
    workers = int(runtime.get("num_workers", 0))
    runtime.setdefault("val_num_workers", min(4, workers))
    runtime.setdefault("official_eval_num_workers", int(runtime["val_num_workers"]))
    runtime.setdefault("val_persistent_workers", False)
    runtime.setdefault("official_eval_persistent_workers", False)
    runtime.setdefault("val_batch_size_per_gpu", int(runtime["batch_size_per_gpu"]))
    runtime.setdefault("official_eval_batch_size_per_gpu", int(runtime["batch_size_per_gpu"]))
    runtime.setdefault("val_prefetch_factor", max(int(runtime.get("prefetch_factor", 2)) // 2, 2))
    runtime.setdefault("official_eval_prefetch_factor", int(runtime["val_prefetch_factor"]))
    runtime.setdefault("official_eval_interval", 1)
    return runtime
def _setup_distributed(config: dict, launcher: str, device: str) -> DistContext:
    if launcher != "ddp":
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        return DistContext(False, 0, 1, 0, dev)
    dist_cfg = config.get("distributed", {})
    backend = dist_cfg.get("backend", "nccl" if torch.cuda.is_available() else "gloo")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
    else:
        dev = torch.device("cpu")
    return DistContext(True, rank, world, local_rank, dev)
def _cleanup_distributed(ctx: DistContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
def _seed_everything(seed: int, rank: int) -> None:
    real_seed = seed + rank
    random.seed(real_seed)
    np.random.seed(real_seed)
    torch.manual_seed(real_seed)
    torch.cuda.manual_seed_all(real_seed)


def _setup_acceleration(runtime: dict, device: torch.device) -> None:
    if device.type != "cuda":
        return
    if bool(runtime.get("cudnn_benchmark", True)) and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
    if bool(runtime.get("tf32", True)):
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True


def _build_dataset(config: dict, split: str) -> OpenLaneDataset:
    data_cfg = config["data"]
    list_path = data_cfg["train_list"] if split == "train" else data_cfg["val_list"]
    return OpenLaneDataset(
        data_root=data_cfg["data_root"],
        list_path=list_path,
        image_size=tuple(data_cfg["image_size"]),
        max_lanes=data_cfg["max_lanes"],
        num_points=data_cfg["num_points"],
        camera_param_policy=data_cfg.get("camera_param_policy", "strict"),
        category_policy=data_cfg.get("category_policy", "preserve_21"),
        preindex_cache=bool(data_cfg.get("preindex_cache", True)),
    )


def _build_loader(config: dict, split: str, ctx: DistContext, runtime: dict) -> tuple[DataLoader, DistributedSampler | None]:
    dataset = _build_dataset(config, split)
    sampler = DistributedSampler(dataset, shuffle=(split == "train")) if ctx.enabled else None
    if split == "train":
        workers = int(runtime["num_workers"])
        persistent = bool(runtime["persistent_workers"])
        batch_size = int(runtime["batch_size_per_gpu"])
        prefetch = int(runtime["prefetch_factor"])
    else:
        workers = int(runtime.get("val_num_workers", runtime["num_workers"]))
        persistent = bool(runtime.get("val_persistent_workers", False))
        batch_size = int(runtime.get("val_batch_size_per_gpu", runtime["batch_size_per_gpu"]))
        prefetch = int(runtime.get("val_prefetch_factor", runtime["prefetch_factor"]))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": (sampler is None and split == "train"),
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": bool(runtime["pin_memory"]),
        "persistent_workers": persistent and workers > 0,
        "collate_fn": openlane_collate,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch
    loader = DataLoader(
        dataset,
        **kwargs,
    )
    return loader, sampler
def _build_model(config: dict, ctx: DistContext):
    model_cfg = config["model"]
    innovation = config["innovation"]
    model = LaneMaster3DNet(
        hidden_dim=model_cfg["hidden_dim"],
        query_count=model_cfg["query_count"],
        num_points=model_cfg["num_points"],
        num_category=int(model_cfg.get("num_category", 21)),
        use_gca=bool(innovation["gca"]["enabled"]),
        backbone_name=model_cfg.get("backbone_name", "resnet101"),
        dynamic_anchor_enabled=bool(innovation["dagp"]["enabled"]),
        dynamic_anchor_delta_scale=float(innovation["dagp"].get("delta_scale", 0.25)),
        y_steps=model_cfg.get("y_steps"),
        anchor_cfg=model_cfg.get("anchor_cfg"),
        iter_reg=int(model_cfg.get("iter_reg", 2)),
        anchor_feat_channels=int(model_cfg.get("anchor_feat_channels", 64)),
        feature_level=int(model_cfg.get("feature_level", 2)),
    ).to(ctx.device)
    if ctx.device.type == "cuda" and bool(config.get("runtime", {}).get("channels_last", True)):
        model = model.to(memory_format=torch.channels_last)
    if ctx.enabled and config.get("distributed", {}).get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    runtime_cfg = config.get("runtime", {})
    if bool(runtime_cfg.get("compile_model", False)) and hasattr(torch, "compile"):
        compile_kwargs = {
            "backend": str(runtime_cfg.get("compile_backend", "inductor")),
            "mode": str(runtime_cfg.get("compile_mode", "reduce-overhead")),
            "dynamic": bool(runtime_cfg.get("compile_dynamic", False)),
        }
        model = torch.compile(model, **compile_kwargs)
    if ctx.enabled:
        kwargs = {
            "find_unused_parameters": bool(config.get("distributed", {}).get("find_unused_parameters", False)),
            "gradient_as_bucket_view": True,
            "static_graph": bool(config.get("runtime", {}).get("ddp_static_graph", True)),
        }
        if ctx.device.type == "cuda":
            model = DDP(model, device_ids=[ctx.local_rank], output_device=ctx.local_rank, **kwargs)
        else:
            model = DDP(model, **kwargs)
    return model
def _unwrap(model):
    return model.module if isinstance(model, DDP) else model
def _build_optimizer(config: dict, model) -> torch.optim.Optimizer:
    optim_cfg = config.get("optim", {})
    lr = float(optim_cfg.get("lr", config["runtime"].get("lr", 1e-4)))
    wd = float(optim_cfg.get("weight_decay", config["runtime"].get("weight_decay", 1e-4)))
    betas = tuple(optim_cfg.get("betas", (0.9, 0.999)))
    runtime_cfg = config.get("runtime", {})
    use_fused = bool(runtime_cfg.get("fused_adamw", True))
    if use_fused and torch.cuda.is_available():
        try:
            return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd, fused=True)
        except TypeError:
            pass
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
def _build_scheduler(config: dict, optimizer: torch.optim.Optimizer, epochs: int):
    sched = config.get("sched", {})
    warmup = int(sched.get("warmup_epochs", 1))
    min_ratio = float(sched.get("min_lr_ratio", 0.05))

    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return max(min_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
def _build_losses(
    config: dict,
) -> tuple[
    LaneSetLoss,
    GeometryConsistencySelfSupervisionLoss,
    float,
    HeteroscedasticLaneUncertaintyLoss,
    dict[str, float | bool],
]:
    num_points = int(config["model"]["num_points"])
    strict_class_check = str(config.get("runtime", {}).get("strict_class_check", "warn"))
    lane_loss = LaneSetLoss(
        anchor_len=num_points,
        gt_anchor_len=num_points,
        anchor_steps=list(range(1, num_points + 1)),
        assign_cfg={"type": "TopKLaneAssigner", "pos_k": 3, "neg_k": 450, "anchor_len": num_points},
        strict_class_check=strict_class_check,
    )
    prior_cfg = config["innovation"]["gpc"]
    prior_loss = GeometryConsistencySelfSupervisionLoss(
        lambda_consistency=float(prior_cfg.get("consistency_weight", 1.0)),
        lambda_smooth=float(prior_cfg.get("smooth_weight", 0.2)),
    )
    unc_cfg = dict(config["innovation"].get("uncertainty", {}))
    unc_cfg.setdefault("enabled", True)
    unc_cfg.setdefault("weight", 0.2)
    unc_cfg.setdefault("logvar_min", -5.0)
    unc_cfg.setdefault("logvar_max", 4.0)
    uncertainty_loss = HeteroscedasticLaneUncertaintyLoss(
        logvar_min=float(unc_cfg["logvar_min"]),
        logvar_max=float(unc_cfg["logvar_max"]),
    )
    return lane_loss, prior_loss, float(prior_cfg.get("weight", 0.3)), uncertainty_loss, unc_cfg
def _match_uncertainty_targets(
    lane_loss: LaneSetLoss,
    proposal: torch.Tensor,
    anchor: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    valid_target = target[target[:, 1] > 0]
    if len(valid_target) == 0:
        return None
    target_short = lane_loss._slice_target(valid_target)
    with torch.no_grad():
        base = anchor if lane_loss.anchor_assign else proposal[:, :5 + lane_loss.anchor_len * 3]
        pos_mask, _, pos_target_idx = lane_loss.assigner.match_proposals_with_targets(base, target_short)
    if int(pos_mask.sum().item()) == 0:
        return None
    pos_index = pos_mask.nonzero(as_tuple=False).squeeze(1)
    return pos_index, target_short[pos_target_idx]
def _split_target_xyz_vis(target: torch.Tensor, num_points: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = target[:, 5:5 + num_points]
    z = target[:, 5 + num_points:5 + num_points * 2]
    vis = target[:, 5 + num_points * 2:5 + num_points * 3]
    return x, z, vis


def _compute_uncertainty_loss(
    output: dict[str, torch.Tensor],
    proposals: list[tuple[torch.Tensor, torch.Tensor]],
    targets: list[torch.Tensor],
    lane_loss: LaneSetLoss,
    uncertainty_loss: HeteroscedasticLaneUncertaintyLoss,
    cached_matches: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
) -> torch.Tensor | None:
    pred_logvar_x = output.get("pred_logvar_x")
    pred_logvar_z = output.get("pred_logvar_z")
    if not isinstance(pred_logvar_x, torch.Tensor) or not isinstance(pred_logvar_z, torch.Tensor):
        return None
    losses: list[torch.Tensor] = []
    num_points = lane_loss.anchor_len
    for batch_idx, (proposal, anchor) in enumerate(proposals):
        matched = None
        if cached_matches is not None and batch_idx < len(cached_matches):
            matched = cached_matches[batch_idx]
        if matched is None:
            matched = _match_uncertainty_targets(lane_loss, proposal, anchor, targets[batch_idx])
        if matched is None:
            continue
        pos_index, pos_target = matched
        pred_pos = proposal.index_select(0, pos_index)
        target_x, target_z, target_vis = _split_target_xyz_vis(pos_target, num_points)
        pred_x = pred_pos[:, 5:5 + num_points]
        pred_z = pred_pos[:, 5 + num_points:5 + num_points * 2]
        logvar_x = pred_logvar_x[batch_idx].index_select(0, pos_index)
        logvar_z = pred_logvar_z[batch_idx].index_select(0, pos_index)
        loss_x = uncertainty_loss(pred_x, target_x, logvar_x, target_vis)
        loss_z = uncertainty_loss(pred_z, target_z, logvar_z, target_vis)
        losses.append(loss_x + loss_z)
    if not losses:
        return pred_logvar_x.new_tensor(0.0)
    return torch.stack(losses).mean()


def _compute_losses(
    output: dict,
    batch: dict,
    lane_loss: LaneSetLoss,
    prior_loss,
    prior_weight: float,
    use_gpc: bool,
    uncertainty_loss: HeteroscedasticLaneUncertaintyLoss,
    uncertainty_weight: float,
    use_uncertainty: bool,
):
    proposals = build_proposals(output)
    targets = build_targets(batch["gt_points"], batch["gt_vis"], batch["lane_valid"], batch.get("gt_category"))
    lane_out = lane_loss(proposals, targets)
    losses = lane_out["losses"].copy()
    total = sum(losses.values())
    ddp_aux = output.get("ddp_aux")
    if isinstance(ddp_aux, torch.Tensor):
        total = total + ddp_aux
    class_index_issues = lane_out.get("class_index_issues", output["pred_points"].new_tensor(0.0))
    if isinstance(class_index_issues, torch.Tensor):
        losses["class_index_issues"] = class_index_issues
    if use_gpc:
        visibility = batch["gt_vis"].new_ones(output["pred_points"].shape[:3])
        prior = prior_loss(output["pred_points"], output["anchor_priors"], visibility)
        losses["loss_gpc"] = prior["loss_total"] * prior_weight
        total = total + losses["loss_gpc"]
    if use_uncertainty:
        cache = lane_out.get("uncertainty_matches")
        cached_matches = cache if isinstance(cache, list) else None
        unc = _compute_uncertainty_loss(
            output,
            proposals,
            targets,
            lane_loss,
            uncertainty_loss,
            cached_matches=cached_matches,
        )
        if unc is not None:
            losses["loss_uncertainty"] = unc * uncertainty_weight
            total = total + losses["loss_uncertainty"]
    return total, losses


def _cast_output_to_float32(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    casted: dict[str, torch.Tensor] = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.dtype.is_floating_point and value.dtype != torch.float32:
            casted[key] = value.float()
        else:
            casted[key] = value
    return casted


def _use_channels_last_on_batch(batch: dict) -> None:
    image = batch.get("image")
    if isinstance(image, torch.Tensor) and image.ndim == 4 and image.device.type == "cuda":
        batch["image"] = image.contiguous(memory_format=torch.channels_last)


def _append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _resolve_amp_dtype(runtime: dict) -> torch.dtype:
    amp_dtype = str(runtime.get("amp_dtype", "float16")).lower()
    if amp_dtype == "float16":
        return torch.float16
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError("runtime.amp_dtype 仅支持 float16/bfloat16")


def _reduce_metric(metric_sum: dict[str, float], sample_count: int, ctx: DistContext) -> dict[str, float]:
    keys = sorted(metric_sum.keys())
    if sample_count <= 0:
        return {k: 0.0 for k in keys}
    if not ctx.enabled:
        denom = float(sample_count)
        return {k: float(metric_sum[k] / denom) for k in keys}
    tensor = torch.tensor([metric_sum[k] for k in keys] + [float(sample_count)], device=ctx.device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    denom = max(float(tensor[-1].item()), 1.0)
    return {k: float(tensor[i].item() / denom) for i, k in enumerate(keys)}


def _weighted_metric_sum(metrics: list[dict[str, float]], sample_counts: list[int]) -> tuple[dict[str, float], int]:
    if not metrics:
        return {}, 0
    if len(metrics) != len(sample_counts):
        raise ValueError("metrics 与 sample_counts 长度必须一致")
    keys = sorted(metrics[0].keys())
    metric_sum = {k: 0.0 for k in keys}
    total_samples = 0
    for metric, count in zip(metrics, sample_counts):
        weight = max(int(count), 0)
        if weight <= 0:
            continue
        total_samples += weight
        for key in keys:
            metric_sum[key] += float(metric.get(key, 0.0)) * float(weight)
    return metric_sum, total_samples


def _eval_one_epoch(model, loader: DataLoader, ctx: DistContext) -> dict[str, float]:
    model.eval()
    metrics = []
    sample_counts = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, ctx.device)
            _use_channels_last_on_batch(batch)
            image = batch.get("image")
            batch_size = int(image.shape[0]) if isinstance(image, torch.Tensor) and image.ndim > 0 else 1
            sample_counts.append(batch_size)
            output = model(batch["image"], project_matrix=build_project_matrix(batch))
            metrics.append(
                evaluate_lane_batch(
                    output["pred_points"],
                    output["pred_scores"],
                    batch["gt_points"],
                    batch["lane_valid"],
                )
            )
    metric_sum, total_samples = _weighted_metric_sum(metrics, sample_counts)
    return _reduce_metric(metric_sum, total_samples, ctx)


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


def _infer_category(logits_row: np.ndarray) -> int:
    if logits_row.shape[0] <= 1:
        return 1
    return int(np.argmax(logits_row[1:]) + 1)


def _build_predictions_for_official(output: dict[str, torch.Tensor], metas: list[dict[str, str]], score_thresh: float):
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
        lane_cates = [_infer_category(logits[idx, lane_idx]) for lane_idx in keep_inds]
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


def _build_main_val_loader(config: dict, runtime: dict) -> DataLoader:
    dataset = _build_dataset(config, "val")
    workers = int(runtime.get("official_eval_num_workers", runtime.get("val_num_workers", runtime["num_workers"])))
    kwargs = {
        "batch_size": int(runtime.get("official_eval_batch_size_per_gpu", runtime["batch_size_per_gpu"])),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": bool(runtime["pin_memory"]),
        "persistent_workers": bool(runtime.get("official_eval_persistent_workers", False)) and workers > 0,
        "collate_fn": openlane_collate,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(
            runtime.get("official_eval_prefetch_factor", runtime.get("val_prefetch_factor", runtime["prefetch_factor"]))
        )
    return DataLoader(dataset, **kwargs)


def _eval_official_one_epoch(model, loader: DataLoader, device: torch.device, score_thresh: float, gt_map_cache: dict[str, dict[str, Any]] | None = None) -> dict[str, float]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    gt_map = gt_map_cache if gt_map_cache is not None else {}
    with torch.no_grad():
        for raw_batch in loader:
            metas = raw_batch["meta"]
            batch = to_device(raw_batch, device)
            _use_channels_last_on_batch(batch)
            output = model(batch["image"], project_matrix=build_project_matrix(batch))
            predictions.extend(_build_predictions_for_official(output, metas, score_thresh))
            for meta in metas:
                file_path, gt = _load_gt_from_meta(meta, gt_map)
                if gt is not None:
                    gt_map[file_path] = gt
    if not gt_map:
        return {"F_score": 0.0}
    return evaluate_openlane_official(predictions, gt_map, prob_th=score_thresh)


def _save_checkpoint(
    model,
    epoch: int,
    metric: dict[str, float],
    official_metric: dict[str, float] | None,
    out_dir: Path,
    best: dict[str, float],
) -> dict[str, float]:
    raw = _unwrap(model).state_dict()
    state = {"epoch": epoch + 1, "model": raw, "metric": metric, "official_metric": official_metric}
    torch.save(state, out_dir / "latest.pth")
    if official_metric is not None and "F_score" in official_metric:
        cur_score = float(official_metric["F_score"])
        score_key = "F_score"
    else:
        cur_score = float(metric.get("f1", 0.0))
        score_key = "f1"
    if cur_score >= float(best.get("score", -1.0)):
        torch.save(state, out_dir / "best.pth")
        return {
            "score": cur_score,
            "score_key": score_key,
            "metric": metric,
            "official_metric": official_metric or {},
        }
    return best


def _log_iter(log_file: Path, rank: int, epoch: int, step: int, total_step: int, losses: dict[str, float], lr: float, data_t: float, iter_t: float):
    payload = {
        "type": "iter",
        "rank": rank,
        "epoch": epoch + 1,
        "iter": step,
        "iter_total": total_step,
        "lr": lr,
        "data_time": data_t,
        "iter_time": iter_t,
        "losses": losses,
    }
    _append_jsonl(log_file, payload)


def train_model(config: dict, work_dir: str, launcher: str = "none", device: str = "cuda") -> dict[str, float]:
    runtime = _runtime(config)
    ctx = _setup_distributed(config, launcher=launcher, device=device)
    _seed_everything(int(runtime["seed"]), ctx.rank)
    _setup_acceleration(runtime, ctx.device)
    out_dir = Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = logs_dir / "metrics.jsonl"
    train_loader, train_sampler = _build_loader(config, "train", ctx, runtime)
    val_loader, val_sampler = _build_loader(config, "val", ctx, runtime)
    model = _build_model(config, ctx)
    optimizer = _build_optimizer(config, model)
    scheduler = _build_scheduler(config, optimizer, int(runtime["epochs"]))
    lane_loss, prior_loss, prior_weight, uncertainty_loss, uncertainty_cfg = _build_losses(config)
    amp_enabled = bool(runtime["amp"]) and ctx.device.type == "cuda"
    autocast_device = "cuda" if ctx.device.type == "cuda" else "cpu"
    amp_dtype = _resolve_amp_dtype(runtime)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    use_gpc = bool(config["innovation"]["gpc"]["enabled"])
    use_uncertainty = bool(uncertainty_cfg["enabled"])
    uncertainty_weight = float(uncertainty_cfg["weight"])
    accum_steps = max(int(runtime["accum_steps"]), 1)
    grad_clip = float(config.get("optim", {}).get("grad_clip", 35.0))
    best_metric = {"score": -1.0, "score_key": "f1", "metric": {}, "official_metric": {}}
    use_official_for_best = bool(runtime.get("official_eval_for_best", runtime.get("official_eval", True)))
    official_eval_interval = int(runtime.get("official_eval_interval", 1))
    score_thresh = float(runtime.get("score_threshold", 0.5))
    main_val_loader = _build_main_val_loader(config, runtime) if (_is_main(ctx) and use_official_for_best) else None
    official_gt_cache: dict[str, dict[str, Any]] | None = {} if main_val_loader is not None else None
    global_step = 0
    non_finite_steps = 0
    for epoch in range(int(runtime["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        model.train()
        last_t = time.time()
        for step, batch in enumerate(train_loader, start=1):
            data_t = time.time() - last_t
            global_step += 1
            batch = to_device(batch, ctx.device)
            _use_channels_last_on_batch(batch)
            autocast_kwargs: dict[str, torch.dtype] = {}
            if amp_enabled and autocast_device == "cuda":
                autocast_kwargs["dtype"] = amp_dtype
            with torch.amp.autocast(autocast_device, enabled=amp_enabled, **autocast_kwargs):
                output = model(batch["image"], project_matrix=build_project_matrix(batch))
            with torch.amp.autocast(autocast_device, enabled=False):
                total, losses = _compute_losses(
                    output=_cast_output_to_float32(output),
                    batch=batch,
                    lane_loss=lane_loss,
                    prior_loss=prior_loss,
                    prior_weight=prior_weight,
                    use_gpc=use_gpc,
                    uncertainty_loss=uncertainty_loss,
                    uncertainty_weight=uncertainty_weight,
                    use_uncertainty=use_uncertainty,
                )
                total = total / accum_steps
            finite_flag = torch.isfinite(total.detach()).to(dtype=torch.float32, device=ctx.device)
            if ctx.enabled:
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            if finite_flag.item() < 0.5:
                non_finite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if _is_main(ctx) and step % int(runtime["log_interval"]) == 0:
                    _append_jsonl(
                        metrics_log,
                        {"type": "warn", "epoch": epoch + 1, "iter": step, "message": "检测到非有限loss，已跳过该step"},
                    )
                last_t = time.time()
                continue
            if step == 1:
                optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            if step % accum_steps == 0 or step == len(train_loader):
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(_unwrap(model).parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            iter_t = time.time() - last_t
            if _is_main(ctx) and step % int(runtime["log_interval"]) == 0:
                loss_num = {k: float(v.detach().item()) for k, v in losses.items()}
                loss_num["non_finite_steps"] = float(non_finite_steps)
                _log_iter(metrics_log, ctx.rank, epoch, step, len(train_loader), loss_num, float(optimizer.param_groups[0]["lr"]), data_t, iter_t)
            last_t = time.time()
        scheduler.step()
        metric = _eval_one_epoch(model, val_loader, ctx)
        if _is_main(ctx):
            official_metric = None
            should_official = _should_run_official_eval(epoch, int(runtime["epochs"]), official_eval_interval)
            if main_val_loader is not None and should_official:
                official_metric = _eval_official_one_epoch(
                    _unwrap(model),
                    main_val_loader,
                    ctx.device,
                    score_thresh,
                    gt_map_cache=official_gt_cache,
                )
            best_metric = _save_checkpoint(model, epoch, metric, official_metric, out_dir, best_metric)
            summary = {
                "type": "epoch",
                "epoch": epoch + 1,
                "metric": metric,
                "official_metric": official_metric,
                "best_metric": best_metric,
            }
            _append_jsonl(metrics_log, summary)
            print(summary)
        if ctx.enabled:
            dist.barrier()
    if _is_main(ctx):
        (out_dir / "metrics.json").write_text(json.dumps(best_metric, indent=2), encoding="utf-8")
    _cleanup_distributed(ctx)
    return best_metric if _is_main(ctx) else {}
