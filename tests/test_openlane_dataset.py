from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from lanemaster3d.data import openlane_dataset as openlane_dataset_mod
from lanemaster3d.data import OpenLaneDataset


def _build_dataset_root(tmp_path: Path, ann: dict, with_image: bool = False) -> Path:
    data_root = tmp_path / "openlane"
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "train.txt").write_text("sample.json\n", encoding="utf-8")
    (data_root / "sample.json").write_text(json.dumps(ann), encoding="utf-8")
    if with_image:
        images_dir = data_root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 32), color=(64, 128, 192)).save(images_dir / "sample.jpg")
    return data_root


def test_camera_param_policy_strict_raise(tmp_path: Path) -> None:
    ann = {
        "file_path": "images/sample.jpg",
        "lane_lines": [{"xyz": [[0.0, 0.1], [5.0, 10.0], [0.0, 0.0]], "visibility": [1, 1], "category": 1}],
    }
    data_root = _build_dataset_root(tmp_path, ann)
    dataset = OpenLaneDataset(
        data_root=str(data_root),
        list_path="train.txt",
        image_size=(128, 256),
        max_lanes=4,
        num_points=6,
        camera_param_policy="strict",
    )
    with pytest.raises(ValueError):
        _ = dataset[0]


def test_missing_list_file_should_raise(tmp_path: Path) -> None:
    data_root = tmp_path / "openlane"
    data_root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        OpenLaneDataset(
            data_root=str(data_root),
            list_path="missing.txt",
            image_size=(128, 256),
            max_lanes=4,
            num_points=6,
        )


def test_camera_param_policy_fallback_default(tmp_path: Path) -> None:
    ann = {
        "file_path": "images/sample.jpg",
        "lane_lines": [{"xyz": [[0.0, 0.1], [5.0, 10.0], [0.0, 0.0]], "visibility": [1, 1], "category": 1}],
    }
    data_root = _build_dataset_root(tmp_path, ann)
    dataset = OpenLaneDataset(
        data_root=str(data_root),
        list_path="train.txt",
        image_size=(128, 256),
        max_lanes=4,
        num_points=6,
        camera_param_policy="fallback",
    )
    sample = dataset[0]
    assert isinstance(sample["cam_extrinsic"], torch.Tensor)
    assert isinstance(sample["cam_intrinsic"], torch.Tensor)
    assert sample["cam_extrinsic"].shape == (4, 4)
    assert sample["cam_intrinsic"].shape == (3, 3)
    assert torch.allclose(sample["cam_extrinsic"], torch.eye(4, dtype=torch.float32))


def test_lane_category_policy_preserve_21(tmp_path: Path) -> None:
    ann = {
        "file_path": "images/sample.jpg",
        "extrinsic": torch.eye(4).tolist(),
        "intrinsic": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        "lane_lines": [{"xyz": [[0.0, 0.1], [5.0, 10.0], [0.0, 0.0]], "visibility": [1, 1], "category": 21}],
    }
    data_root = _build_dataset_root(tmp_path, ann)
    dataset = OpenLaneDataset(
        data_root=str(data_root),
        list_path="train.txt",
        image_size=(128, 256),
        max_lanes=4,
        num_points=6,
        category_policy="preserve_21",
    )
    sample = dataset[0]
    assert float(sample["gt_category"][0].item()) == 21.0


def test_lane_category_policy_legacy_map(tmp_path: Path) -> None:
    ann = {
        "file_path": "images/sample.jpg",
        "extrinsic": torch.eye(4).tolist(),
        "intrinsic": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        "lane_lines": [{"xyz": [[0.0, 0.1], [5.0, 10.0], [0.0, 0.0]], "visibility": [1, 1], "category": 21}],
    }
    data_root = _build_dataset_root(tmp_path, ann)
    dataset = OpenLaneDataset(
        data_root=str(data_root),
        list_path="train.txt",
        image_size=(128, 256),
        max_lanes=4,
        num_points=6,
        category_policy="legacy_map_21_to_20",
    )
    sample = dataset[0]
    assert float(sample["gt_category"][0].item()) == 20.0


def test_empty_lane_should_not_be_marked_valid(tmp_path: Path) -> None:
    ann = {
        "file_path": "images/sample.jpg",
        "extrinsic": torch.eye(4).tolist(),
        "intrinsic": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        "lane_lines": [{"xyz": [[], [], []], "category": 1}],
    }
    data_root = _build_dataset_root(tmp_path, ann)
    dataset = OpenLaneDataset(
        data_root=str(data_root),
        list_path="train.txt",
        image_size=(128, 256),
        max_lanes=4,
        num_points=6,
    )
    sample = dataset[0]
    assert float(sample["lane_valid"][0].item()) == 0.0
    assert float(sample["gt_vis"][0].sum().item()) == 0.0


def test_lane_with_zero_visibility_should_be_marked_invalid(tmp_path: Path) -> None:
    ann = {
        "file_path": "images/sample.jpg",
        "extrinsic": torch.eye(4).tolist(),
        "intrinsic": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        "lane_lines": [{"xyz": [[0.0, 0.1], [5.0, 10.0], [0.0, 0.0]], "visibility": [0.0, 0.0], "category": 1}],
    }
    data_root = _build_dataset_root(tmp_path, ann)
    dataset = OpenLaneDataset(
        data_root=str(data_root),
        list_path="train.txt",
        image_size=(128, 256),
        max_lanes=4,
        num_points=6,
    )
    sample = dataset[0]
    assert float(sample["lane_valid"][0].item()) == 0.0


def test_dataset_preindex_is_lightweight(tmp_path: Path) -> None:
    ann = {
        "file_path": "images/sample.jpg",
        "extrinsic": torch.eye(4).tolist(),
        "intrinsic": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        "lane_lines": [{"xyz": [[0.0, 0.1], [5.0, 10.0], [0.0, 0.0]], "visibility": [1.0, 1.0], "category": 1}],
    }
    data_root = _build_dataset_root(tmp_path, ann)
    dataset = OpenLaneDataset(
        data_root=str(data_root),
        list_path="train.txt",
        image_size=(128, 256),
        max_lanes=4,
        num_points=6,
        camera_param_policy="strict",
    )
    entry = dataset.sample_entries[0]
    assert not hasattr(entry, "ann")
    assert entry.ann_path is not None
    sample = dataset[0]
    assert float(sample["lane_valid"][0].item()) == 1.0


def test_dataset_preindex_cache_can_be_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ann = {
        "file_path": "images/sample.jpg",
        "extrinsic": torch.eye(4).tolist(),
        "intrinsic": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        "lane_lines": [{"xyz": [[0.0, 0.1], [5.0, 10.0], [0.0, 0.0]], "visibility": [1.0, 1.0], "category": 1}],
    }
    data_root = _build_dataset_root(tmp_path, ann)
    dataset = OpenLaneDataset(
        data_root=str(data_root),
        list_path="train.txt",
        image_size=(128, 256),
        max_lanes=4,
        num_points=6,
        camera_param_policy="strict",
        preindex_cache=True,
    )
    cache_path = data_root / ".cache" / "openlane_preindex_train.json"
    assert cache_path.exists()

    def _raise_if_called(*_args, **_kwargs):
        raise RuntimeError("不应在缓存命中时重新构建预索引")

    monkeypatch.setattr(openlane_dataset_mod, "_build_sample_entry", _raise_if_called)
    dataset2 = OpenLaneDataset(
        data_root=str(data_root),
        list_path="train.txt",
        image_size=(128, 256),
        max_lanes=4,
        num_points=6,
        camera_param_policy="strict",
        preindex_cache=True,
    )
    sample = dataset2[0]
    assert float(sample["lane_valid"][0].item()) == 1.0


def test_dataset_preindex_cache_should_invalidate_when_list_content_changes(tmp_path: Path) -> None:
    data_root = tmp_path / "openlane"
    data_root.mkdir(parents=True, exist_ok=True)
    ann = {
        "file_path": "images/sample.jpg",
        "extrinsic": torch.eye(4).tolist(),
        "intrinsic": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        "lane_lines": [{"xyz": [[0.0, 0.1], [5.0, 10.0], [0.0, 0.0]], "visibility": [1.0, 1.0], "category": 1}],
    }
    (data_root / "sample_a.json").write_text(json.dumps(ann), encoding="utf-8")
    (data_root / "sample_b.json").write_text(json.dumps(ann), encoding="utf-8")
    (data_root / "train.txt").write_text("sample_a.json\n", encoding="utf-8")

    dataset_a = OpenLaneDataset(
        data_root=str(data_root),
        list_path="train.txt",
        image_size=(128, 256),
        max_lanes=4,
        num_points=6,
        camera_param_policy="strict",
        preindex_cache=True,
    )
    assert dataset_a.sample_entries[0].item == "sample_a.json"

    (data_root / "train.txt").write_text("sample_b.json\n", encoding="utf-8")
    dataset_b = OpenLaneDataset(
        data_root=str(data_root),
        list_path="train.txt",
        image_size=(128, 256),
        max_lanes=4,
        num_points=6,
        camera_param_policy="strict",
        preindex_cache=True,
    )
    assert dataset_b.sample_entries[0].item == "sample_b.json"


def test_dataset_timing_probe_should_emit_sample_timing(tmp_path: Path) -> None:
    ann = {
        "file_path": "images/sample.jpg",
        "extrinsic": torch.eye(4).tolist(),
        "intrinsic": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        "lane_lines": [{"xyz": [[0.0, 0.1], [5.0, 10.0], [0.0, 0.0]], "visibility": [1.0, 1.0], "category": 1}],
    }
    data_root = _build_dataset_root(tmp_path, ann)
    dataset = OpenLaneDataset(
        data_root=str(data_root),
        list_path="train.txt",
        image_size=(128, 256),
        max_lanes=4,
        num_points=6,
        camera_param_policy="strict",
        enable_timing_probe=True,
    )
    sample = dataset[0]
    assert "sample_timing" in sample
    assert sample["sample_timing"].shape == (5,)
    assert bool((sample["sample_timing"] >= 0).all().item())

    batch = openlane_dataset_mod.openlane_collate([sample, sample])
    assert "sample_timing" in batch
    assert batch["sample_timing"].shape == (2, 5)


def test_dataset_image_backend_invalid_should_raise(tmp_path: Path) -> None:
    ann = {
        "file_path": "images/sample.jpg",
        "extrinsic": torch.eye(4).tolist(),
        "intrinsic": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        "lane_lines": [{"xyz": [[0.0, 0.1], [5.0, 10.0], [0.0, 0.0]], "visibility": [1.0, 1.0], "category": 1}],
    }
    data_root = _build_dataset_root(tmp_path, ann)
    with pytest.raises(ValueError):
        OpenLaneDataset(
            data_root=str(data_root),
            list_path="train.txt",
            image_size=(128, 256),
            max_lanes=4,
            num_points=6,
            camera_param_policy="strict",
            image_backend="unknown",
        )


def test_dataset_image_backend_cv2_should_keep_image_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ann = {
        "file_path": "images/sample.jpg",
        "extrinsic": torch.eye(4).tolist(),
        "intrinsic": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        "lane_lines": [{"xyz": [[0.0, 0.1], [5.0, 10.0], [0.0, 0.0]], "visibility": [1.0, 1.0], "category": 1}],
    }
    data_root = _build_dataset_root(tmp_path, ann, with_image=True)
    monkeypatch.setattr(openlane_dataset_mod, "cv2", None)
    dataset = OpenLaneDataset(
        data_root=str(data_root),
        list_path="train.txt",
        image_size=(32, 64),
        max_lanes=4,
        num_points=6,
        camera_param_policy="strict",
        image_backend="cv2",
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 32, 64)
    assert float(sample["image"].sum().item()) > 0.0
