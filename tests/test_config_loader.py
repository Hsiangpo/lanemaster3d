from __future__ import annotations

from pathlib import Path

import pytest

from lanemaster3d.config_loader import load_python_config, validate_config


def test_load_python_config_success(tmp_path: Path) -> None:
    cfg_path = tmp_path / "demo_config.py"
    cfg_path.write_text(
        "CONFIG = {\n"
        "  'project': {'name': 'demo'},\n"
        "  'data': {'data_root': './data', 'category_policy': 'preserve_21'},\n"
        "  'model': {'query_count': 24, 'hidden_dim': 64, 'num_points': 10, 'num_category': 22},\n"
        "  'runtime': {'seed': 3407, 'epochs': 1, 'num_workers': 0},\n"
        "  'innovation': {\n"
        "    'gca': {'enabled': True},\n"
        "    'dagp': {'enabled': True, 'delta_scale': 0.25},\n"
        "    'gpc': {'enabled': True, 'weight': 0.3},\n"
        "  },\n"
        "}\n",
        encoding="utf-8",
    )
    config = validate_config(load_python_config(str(cfg_path)))
    assert config["project"]["name"] == "demo"
    assert config["distributed"]["backend"] in {"nccl", "gloo"}
    assert config["innovation"]["dagp"]["enabled"] is True


def test_validate_config_missing_key() -> None:
    with pytest.raises(ValueError):
        validate_config({"project": {}, "data": {}, "runtime": {}})


def test_validate_config_reject_preserve_21_with_21_classes() -> None:
    cfg = {
        "project": {"name": "demo"},
        "data": {
            "data_root": "./data",
            "train_list": "train.txt",
            "val_list": "val.txt",
            "image_size": [720, 960],
            "max_lanes": 24,
            "num_points": 20,
            "category_policy": "preserve_21",
        },
        "model": {"query_count": 24, "hidden_dim": 64, "num_points": 20, "num_category": 21},
        "runtime": {"seed": 3407, "epochs": 1, "num_workers": 0},
        "innovation": {"gca": {"enabled": True}, "dagp": {"enabled": True}, "gpc": {"enabled": True}},
    }
    with pytest.raises(ValueError, match="num_category"):
        validate_config(cfg)


def test_validate_config_reject_legacy_map_with_20_classes() -> None:
    cfg = {
        "project": {"name": "demo"},
        "data": {
            "data_root": "./data",
            "train_list": "train.txt",
            "val_list": "val.txt",
            "image_size": [720, 960],
            "max_lanes": 24,
            "num_points": 20,
            "category_policy": "legacy_map_21_to_20",
        },
        "model": {"query_count": 24, "hidden_dim": 64, "num_points": 20, "num_category": 20},
        "runtime": {"seed": 3407, "epochs": 1, "num_workers": 0},
        "innovation": {"gca": {"enabled": True}, "dagp": {"enabled": True}, "gpc": {"enabled": True}},
    }
    with pytest.raises(ValueError, match="num_category"):
        validate_config(cfg)


def test_validate_config_reject_invalid_strict_class_check() -> None:
    cfg = {
        "project": {"name": "demo"},
        "data": {
            "data_root": "./data",
            "train_list": "train.txt",
            "val_list": "val.txt",
            "image_size": [720, 960],
            "max_lanes": 24,
            "num_points": 20,
            "category_policy": "legacy_map_21_to_20",
        },
        "model": {"query_count": 24, "hidden_dim": 64, "num_points": 20, "num_category": 21},
        "runtime": {"seed": 3407, "epochs": 1, "num_workers": 0, "strict_class_check": "bad_mode"},
        "innovation": {"gca": {"enabled": True}, "dagp": {"enabled": True}, "gpc": {"enabled": True}},
    }
    with pytest.raises(ValueError, match="strict_class_check"):
        validate_config(cfg)


def test_validate_config_reject_invalid_official_eval_interval() -> None:
    cfg = {
        "project": {"name": "demo"},
        "data": {
            "data_root": "./data",
            "train_list": "train.txt",
            "val_list": "val.txt",
            "image_size": [720, 960],
            "max_lanes": 24,
            "num_points": 20,
            "category_policy": "legacy_map_21_to_20",
        },
        "model": {"query_count": 24, "hidden_dim": 64, "num_points": 20, "num_category": 21},
        "runtime": {"seed": 3407, "epochs": 1, "num_workers": 0, "official_eval_interval": 0},
        "innovation": {"gca": {"enabled": True}, "dagp": {"enabled": True}, "gpc": {"enabled": True}},
    }
    with pytest.raises(ValueError, match="official_eval_interval"):
        validate_config(cfg)


def test_validate_config_reject_invalid_amp_dtype() -> None:
    cfg = {
        "project": {"name": "demo"},
        "data": {
            "data_root": "./data",
            "train_list": "train.txt",
            "val_list": "val.txt",
            "image_size": [720, 960],
            "max_lanes": 24,
            "num_points": 20,
            "category_policy": "legacy_map_21_to_20",
        },
        "model": {"query_count": 24, "hidden_dim": 64, "num_points": 20, "num_category": 21},
        "runtime": {"seed": 3407, "epochs": 1, "num_workers": 0, "amp_dtype": "fp8"},
        "innovation": {"gca": {"enabled": True}, "dagp": {"enabled": True}, "gpc": {"enabled": True}},
    }
    with pytest.raises(ValueError, match="amp_dtype"):
        validate_config(cfg)
