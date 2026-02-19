from __future__ import annotations

import importlib.util
import copy
from pathlib import Path
from types import ModuleType
from typing import Any, Dict


REQUIRED_TOP_LEVEL_KEYS = {
    "project",
    "data",
    "model",
    "runtime",
    "innovation",
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "distributed": {
        "backend": "nccl",
        "sync_bn": True,
        "find_unused_parameters": False,
    },
    "optim": {
        "type": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "betas": [0.9, 0.999],
        "grad_clip": 35.0,
    },
    "sched": {
        "type": "cosine",
        "warmup_epochs": 1,
        "min_lr_ratio": 0.05,
    },
    "runtime": {
        "amp": True,
        "amp_dtype": "float16",
        "accum_steps": 1,
        "log_interval": 20,
        "strict_class_check": "warn",
        "official_eval_interval": 1,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "batch_size_per_gpu": 4,
    },
    "innovation": {
        "gca": {"enabled": True},
        "dagp": {"enabled": True, "delta_scale": 0.25},
        "gpc": {"enabled": True, "weight": 0.3, "consistency_weight": 1.0, "smooth_weight": 0.2},
        "uncertainty": {"enabled": True, "weight": 0.2, "logvar_min": -5.0, "logvar_max": 4.0},
    },
}


def _load_module(config_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("lanemaster3d_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载配置文件: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_python_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    module = _load_module(path)
    config = getattr(module, "CONFIG", None)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须定义 dict 类型 CONFIG: {path}")
    return config


def _merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if key in dst and isinstance(dst[key], dict) and isinstance(value, dict):
            _merge_dict(dst[key], value)
        else:
            dst[key] = value
    return dst


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_CONFIG)
    _merge_dict(merged, config)
    return merged


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    missing_keys = REQUIRED_TOP_LEVEL_KEYS - set(config.keys())
    if missing_keys:
        keys = ", ".join(sorted(missing_keys))
        raise ValueError(f"配置缺少必要字段: {keys}")
    merged = normalize_config(config)
    if not isinstance(merged["innovation"]["gca"], dict):
        raise ValueError("innovation.gca 必须是字典结构")
    if not isinstance(merged["innovation"]["dagp"], dict):
        raise ValueError("innovation.dagp 必须是字典结构")
    if not isinstance(merged["innovation"]["gpc"], dict):
        raise ValueError("innovation.gpc 必须是字典结构")
    if not isinstance(merged["innovation"]["uncertainty"], dict):
        raise ValueError("innovation.uncertainty 必须是字典结构")
    if "enabled" not in merged["innovation"]["gca"]:
        raise ValueError("innovation.gca 必须包含 enabled 字段")
    if "enabled" not in merged["innovation"]["dagp"]:
        raise ValueError("innovation.dagp 必须包含 enabled 字段")
    if "enabled" not in merged["innovation"]["gpc"]:
        raise ValueError("innovation.gpc 必须包含 enabled 字段")
    if "enabled" not in merged["innovation"]["uncertainty"]:
        raise ValueError("innovation.uncertainty 必须包含 enabled 字段")
    category_policy = merged["data"].get("category_policy", "preserve_21")
    num_category = int(merged["model"].get("num_category", 21))
    strict_class_check = merged["runtime"].get("strict_class_check", "warn")
    amp_dtype = str(merged["runtime"].get("amp_dtype", "float16"))
    official_eval_interval = int(merged["runtime"].get("official_eval_interval", 1))
    if category_policy == "preserve_21" and num_category < 22:
        raise ValueError("当 data.category_policy=preserve_21 时，model.num_category 必须 >= 22")
    if category_policy == "legacy_map_21_to_20" and num_category < 21:
        raise ValueError("当 data.category_policy=legacy_map_21_to_20 时，model.num_category 必须 >= 21")
    if strict_class_check not in {"off", "warn", "raise"}:
        raise ValueError("runtime.strict_class_check 仅支持 off/warn/raise")
    if amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("runtime.amp_dtype 仅支持 float16/bfloat16")
    if official_eval_interval < 1:
        raise ValueError("runtime.official_eval_interval 必须 >= 1")
    return merged
