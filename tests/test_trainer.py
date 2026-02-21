from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from lanemaster3d.engine import evaluator, trainer
from lanemaster3d.engine.trainer import (
    DistContext,
    _average_metric_list,
    _compute_uncertainty_loss,
    _eval_one_epoch,
    _load_gt_from_meta,
    _resolve_quick_eval_plan,
    _save_checkpoint,
    _should_run_official_eval,
)
from lanemaster3d.models.losses.heteroscedastic import HeteroscedasticLaneUncertaintyLoss


def _dummy_model() -> torch.nn.Module:
    return torch.nn.Linear(4, 2)


def _cpu_ctx() -> DistContext:
    return DistContext(enabled=False, rank=0, world_size=1, local_rank=0, device=torch.device("cpu"))


def _dummy_batch(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "image": torch.zeros(batch_size, 3, 8, 8, dtype=torch.float32),
        "gt_points": torch.zeros(batch_size, 1, 2, 3, dtype=torch.float32),
        "lane_valid": torch.zeros(batch_size, 1, dtype=torch.float32),
    }


class _FakeEvalModel:
    def eval(self) -> "_FakeEvalModel":
        return self

    def __call__(self, image: torch.Tensor, project_matrix=None) -> dict[str, torch.Tensor]:
        batch_size = int(image.shape[0])
        return {
            "pred_points": torch.zeros(batch_size, 1, 2, 3, dtype=torch.float32),
            "pred_scores": torch.zeros(batch_size, 1, dtype=torch.float32),
        }


def test_save_checkpoint_prefers_official_metric(tmp_path: Path) -> None:
    model = _dummy_model()
    best = {"score": -1.0, "score_key": "f1", "metric": {}, "official_metric": {}}
    metric = {"f1": 0.88}
    official = {"F_score": 0.61}
    result = _save_checkpoint(model, epoch=0, metric=metric, official_metric=official, out_dir=tmp_path, best=best)
    assert result["score_key"] == "F_score"
    assert abs(float(result["score"]) - 0.61) < 1e-6
    assert (tmp_path / "latest.pth").exists()
    assert (tmp_path / "best.pth").exists()


def test_save_checkpoint_fallback_to_quick_metric(tmp_path: Path) -> None:
    model = _dummy_model()
    best = {"score": 0.61, "score_key": "F_score", "metric": {}, "official_metric": {"F_score": 0.61}}
    metric = {"f1": 0.72}
    result = _save_checkpoint(model, epoch=1, metric=metric, official_metric=None, out_dir=tmp_path, best=best)
    assert result["score_key"] == "f1"
    assert abs(float(result["score"]) - 0.72) < 1e-6


def test_should_run_official_eval_interval_and_last_epoch() -> None:
    assert _should_run_official_eval(epoch=0, total_epochs=10, interval=3) is False
    assert _should_run_official_eval(epoch=2, total_epochs=10, interval=3) is True
    assert _should_run_official_eval(epoch=8, total_epochs=10, interval=3) is True
    assert _should_run_official_eval(epoch=9, total_epochs=10, interval=3) is True


def test_load_gt_from_meta_reuses_cache(tmp_path: Path) -> None:
    ann_path = tmp_path / "sample.json"
    ann_path.write_text(json.dumps({"lane_lines": []}), encoding="utf-8")
    meta = {"file_path": "seg/sample.jpg", "ann_path": str(ann_path)}
    cache: dict[str, dict] = {}
    file_path, gt = _load_gt_from_meta(meta, cache)
    assert file_path == "seg/sample.jpg"
    assert gt is not None
    ann_path.unlink()
    file_path2, gt2 = _load_gt_from_meta(meta, cache)
    assert file_path2 == "seg/sample.jpg"
    assert gt2 is not None


def test_average_metric_list_returns_exact_mean() -> None:
    metrics = [{"f1": 0.5, "precision": 0.2}, {"f1": 1.0, "precision": 0.4}, {"f1": 1.0, "precision": 1.0}]
    mean = _average_metric_list(metrics)
    assert abs(mean["f1"] - (2.5 / 3.0)) < 1e-6
    assert abs(mean["precision"] - (1.6 / 3.0)) < 1e-6


def test_compute_uncertainty_loss_prefers_cached_matches(monkeypatch) -> None:
    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("有缓存匹配时不应重复匹配")

    monkeypatch.setattr(trainer, "_match_uncertainty_targets", _raise_if_called)
    lane_loss = SimpleNamespace(anchor_len=2, anchor_assign=True, assigner=None)
    proposal = torch.zeros(2, 11, dtype=torch.float32)
    proposal[0, 5:7] = torch.tensor([0.1, 0.2])
    proposal[0, 7:9] = torch.tensor([0.0, 0.1])
    output = {
        "pred_logvar_x": torch.zeros(1, 2, 2, dtype=torch.float32),
        "pred_logvar_z": torch.zeros(1, 2, 2, dtype=torch.float32),
    }
    pos_index = torch.tensor([0], dtype=torch.long)
    pos_target = torch.zeros(1, 11, dtype=torch.float32)
    pos_target[:, 5:7] = torch.tensor([[0.2, 0.3]], dtype=torch.float32)
    pos_target[:, 7:9] = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    pos_target[:, 9:11] = 1.0
    unc = _compute_uncertainty_loss(
        output=output,
        proposals=[(proposal, proposal[:, :11])],
        targets=[pos_target],
        lane_loss=lane_loss,
        uncertainty_loss=HeteroscedasticLaneUncertaintyLoss(),
        cached_matches=[(pos_index, pos_target)],
    )
    assert unc is not None
    assert torch.isfinite(unc)


def test_eval_one_epoch_uses_sample_weighted_mean(monkeypatch) -> None:
    metrics = [
        {"f1": 100.0, "precision": 100.0, "recall": 100.0, "x_error": 1.0, "z_error": 1.0},
        {"f1": 0.0, "precision": 0.0, "recall": 0.0, "x_error": 5.0, "z_error": 5.0},
    ]

    def _fake_eval_lane_batch(*_args, **_kwargs):
        return metrics.pop(0)

    monkeypatch.setattr(trainer, "evaluate_lane_batch", _fake_eval_lane_batch)
    monkeypatch.setattr(trainer, "build_project_matrix", lambda _batch: None)
    monkeypatch.setattr(trainer, "to_device", lambda batch, _device: batch)
    model = _FakeEvalModel()
    loader = [_dummy_batch(1), _dummy_batch(3)]

    result = _eval_one_epoch(model, loader, _cpu_ctx())

    assert abs(result["f1"] - 25.0) < 1e-6
    assert abs(result["precision"] - 25.0) < 1e-6
    assert abs(result["x_error"] - 4.0) < 1e-6


def test_average_quick_metrics_supports_sample_weights() -> None:
    metrics = [
        {"f1": 100.0, "precision": 100.0, "recall": 100.0, "x_error": 1.0, "z_error": 1.0},
        {"f1": 0.0, "precision": 0.0, "recall": 0.0, "x_error": 5.0, "z_error": 5.0},
    ]
    weighted = evaluator._average_quick_metrics(metrics, sample_counts=[1, 3])
    assert abs(weighted["f1"] - 25.0) < 1e-6
    assert abs(weighted["precision"] - 25.0) < 1e-6
    assert abs(weighted["x_error"] - 4.0) < 1e-6


def test_runtime_defaults_should_disable_data_probe() -> None:
    runtime = trainer._runtime({"runtime": {"seed": 1, "num_workers": 2, "batch_size": 4}})
    assert bool(runtime["data_probe"]) is False
    assert bool(runtime["quick_eval_distributed"]) is False
    assert int(runtime["quick_eval_num_workers"]) == 0


def test_resolve_ddp_timeout_seconds_prefers_distributed_and_clamps_minimum() -> None:
    timeout = trainer._resolve_ddp_timeout_seconds(
        {
            "distributed": {"timeout_seconds": 300},
            "runtime": {"ddp_timeout_seconds": 1800},
        }
    )
    assert int(timeout) == 600


def test_resolve_ddp_timeout_seconds_should_fallback_to_runtime_then_default() -> None:
    timeout_runtime = trainer._resolve_ddp_timeout_seconds({"runtime": {"ddp_timeout_seconds": 1800}})
    timeout_default = trainer._resolve_ddp_timeout_seconds({})
    assert int(timeout_runtime) == 1800
    assert int(timeout_default) == 7200


def test_resolve_quick_eval_plan_should_skip_non_main_rank_when_disabled() -> None:
    runtime = {"quick_eval_distributed": False}
    ctx = DistContext(enabled=True, rank=1, world_size=2, local_rank=1, device=torch.device("cpu"))
    plan = _resolve_quick_eval_plan(runtime, ctx)
    assert bool(plan.should_run) is False
    assert bool(plan.use_distributed) is False


def test_resolve_quick_eval_plan_should_run_main_rank_locally_when_disabled() -> None:
    runtime = {"quick_eval_distributed": False}
    ctx = DistContext(enabled=True, rank=0, world_size=2, local_rank=0, device=torch.device("cpu"))
    plan = _resolve_quick_eval_plan(runtime, ctx)
    assert bool(plan.should_run) is True
    assert bool(plan.use_distributed) is False


def test_resolve_quick_eval_plan_should_run_all_ranks_when_enabled() -> None:
    runtime = {"quick_eval_distributed": True}
    rank0 = DistContext(enabled=True, rank=0, world_size=2, local_rank=0, device=torch.device("cpu"))
    rank1 = DistContext(enabled=True, rank=1, world_size=2, local_rank=1, device=torch.device("cpu"))
    plan0 = _resolve_quick_eval_plan(runtime, rank0)
    plan1 = _resolve_quick_eval_plan(runtime, rank1)
    assert bool(plan0.should_run) is True
    assert bool(plan0.use_distributed) is True
    assert bool(plan1.should_run) is True
    assert bool(plan1.use_distributed) is True


def test_summarize_sample_timing_should_return_mean_and_max() -> None:
    sample_timing = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 1.0],
            [0.3, 0.4, 0.5, 0.6, 2.0],
        ],
        dtype=torch.float32,
    )
    summary = trainer._summarize_sample_timing(sample_timing)
    assert abs(summary["sample_time_ann_mean"] - 0.2) < 1e-6
    assert abs(summary["sample_time_ann_max"] - 0.3) < 1e-6
    assert abs(summary["sample_time_sample_total_mean"] - 1.5) < 1e-6
    assert abs(summary["sample_time_sample_total_max"] - 2.0) < 1e-6


def test_log_iter_should_write_stage_times_and_data_probe(tmp_path: Path) -> None:
    log_file = tmp_path / "metrics.jsonl"
    trainer._log_iter(
        log_file=log_file,
        rank=0,
        epoch=0,
        step=20,
        total_step=100,
        losses={"loss_total": 1.0},
        lr=1e-4,
        data_t=0.01,
        iter_t=0.05,
        stage_times={"time_forward": 0.02},
        data_probe={"sample_time_image_mean": 0.003},
    )
    row = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert row["stage_times"]["time_forward"] == 0.02
    assert row["data_probe"]["sample_time_image_mean"] == 0.003


def test_build_dataset_should_pass_image_backend_and_data_probe(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _DummyDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(trainer, "OpenLaneDataset", _DummyDataset)
    cfg = {
        "data": {
            "data_root": "./data/OpenLane",
            "train_list": "data_lists/training.txt",
            "val_list": "data_lists/validation.txt",
            "image_size": [720, 960],
            "max_lanes": 24,
            "num_points": 20,
            "image_backend": "cv2",
        }
    }
    runtime = {"data_probe": True}
    _ = trainer._build_dataset(cfg, "train", runtime)
    assert captured["image_backend"] == "cv2"
    assert captured["enable_timing_probe"] is True
