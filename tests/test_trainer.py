from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from lanemaster3d.engine import trainer
from lanemaster3d.engine.trainer import _average_metric_list, _compute_uncertainty_loss, _load_gt_from_meta, _save_checkpoint, _should_run_official_eval
from lanemaster3d.models.losses.heteroscedastic import HeteroscedasticLaneUncertaintyLoss


def _dummy_model() -> torch.nn.Module:
    return torch.nn.Linear(4, 2)


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
