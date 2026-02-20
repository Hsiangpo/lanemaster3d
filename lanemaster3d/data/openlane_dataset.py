from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

DEFAULT_EXTRINSIC = np.eye(4, dtype=np.float32)
DEFAULT_INTRINSIC = np.array(
    [
        [1000.0, 0.0, 960.0],
        [0.0, 1000.0, 540.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True, slots=True)
class SampleEntry:
    item: str
    ann_path: Path | None
    img_path: Path | None
    file_path: str


def _read_list_file(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"数据列表不存在: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _resolve_annotation_path(data_root: Path, item: str) -> Path | None:
    candidates = [
        data_root / item,
        data_root / f"{item}.json",
        data_root / "lane3d_1000" / item,
        data_root / "lane3d_1000" / f"{item}.json",
        data_root / "annotations" / item,
        data_root / "annotations" / f"{item}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_json(path: Path | None) -> Dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_image_path(data_root: Path, ann: Dict, item: str) -> Path | None:
    keys = ["file_path", "raw_file", "img_path", "image_path"]
    for key in keys:
        value = ann.get(key)
        if isinstance(value, str):
            candidate = data_root / value
            if candidate.exists():
                return candidate
    candidates = [
        data_root / "images" / item.replace(".json", ".jpg"),
        data_root / "images" / item.replace(".json", ".png"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_image(path: Path | None, image_size: Tuple[int, int], image_backend: str) -> tuple[Tensor, tuple[int, int]]:
    height, width = image_size
    if path is None or not path.exists():
        return torch.zeros(3, height, width, dtype=torch.float32), (height, width)
    if image_backend == "cv2" and cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            src_h, src_w = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            arr = image.astype(np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), (int(src_h), int(src_w))
    image = Image.open(path).convert("RGB")
    src_w, src_h = image.size
    image = image.resize((width, height))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), (int(src_h), int(src_w))


def _resample_lane(points: np.ndarray, target_points: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((target_points, 3), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, target_points, axis=0).astype(np.float32)
    src = np.linspace(0.0, 1.0, num=len(points))
    dst = np.linspace(0.0, 1.0, num=target_points)
    out = np.stack([np.interp(dst, src, points[:, i]) for i in range(3)], axis=-1)
    return out.astype(np.float32)


def _extract_lane_points(lane_obj: Dict | List) -> np.ndarray:
    if isinstance(lane_obj, dict):
        xyz = lane_obj.get("xyz")
        if isinstance(xyz, list) and len(xyz) == 3:
            return np.stack([np.asarray(xyz[i], dtype=np.float32) for i in range(3)], axis=-1)
        points = lane_obj.get("points") or lane_obj.get("lane")
        if isinstance(points, list) and points:
            return np.asarray(points, dtype=np.float32)
    if isinstance(lane_obj, list) and lane_obj:
        return np.asarray(lane_obj, dtype=np.float32)
    return np.zeros((0, 3), dtype=np.float32)


def _lane_category(lane_obj: Dict | List, category_policy: str) -> float:
    if isinstance(lane_obj, dict):
        value = lane_obj.get("category", 1)
        if isinstance(value, (int, float)) and value > 0:
            cate = int(value)
            if cate == 21 and category_policy == "legacy_map_21_to_20":
                cate = 20
            return float(max(cate, 1))
    return 1.0


def _parse_lanes(ann: Dict, max_lanes: int, num_points: int, category_policy: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    points = torch.zeros(max_lanes, num_points, 3, dtype=torch.float32)
    vis = torch.zeros(max_lanes, num_points, dtype=torch.float32)
    lane_valid = torch.zeros(max_lanes, dtype=torch.float32)
    lane_category = torch.zeros(max_lanes, dtype=torch.float32)
    lanes = ann.get("lane_lines") or ann.get("lanes") or ann.get("lane3d") or []
    valid_count = min(len(lanes), max_lanes)
    for idx in range(valid_count):
        lane_pts = _extract_lane_points(lanes[idx])
        sampled = _resample_lane(lane_pts, num_points)
        points[idx] = torch.from_numpy(sampled)
        lane_category[idx] = _lane_category(lanes[idx], category_policy)
        if isinstance(lanes[idx], dict) and isinstance(lanes[idx].get("visibility"), list):
            vis_vals = np.asarray(lanes[idx]["visibility"], dtype=np.float32)
            vis_sampled = _resample_lane(np.stack([vis_vals, vis_vals, vis_vals], axis=-1), num_points)[:, 0]
            vis[idx] = torch.from_numpy(np.clip(vis_sampled, 0.0, 1.0))
            lane_valid[idx] = 1.0 if (len(lane_pts) > 0 and float(vis[idx].sum().item()) > 0.0) else 0.0
        else:
            vis[idx].fill_(1.0 if len(lane_pts) > 0 else 0.0)
            lane_valid[idx] = 1.0 if len(lane_pts) > 0 else 0.0
    return points, vis, lane_valid, lane_category


def _parse_file_path(ann: Dict, item: str) -> str:
    for key in ["file_path", "raw_file", "image_path", "img_path"]:
        value = ann.get(key)
        if isinstance(value, str) and value:
            return value
    item_name = item.replace("\\", "/")
    if item_name.endswith(".json"):
        return item_name[:-5] + ".jpg"
    return item_name


def _parse_src_image_hw(ann: Dict, default_hw: tuple[int, int]) -> tuple[int, int]:
    pairs = [("img_h", "img_w"), ("height", "width"), ("h", "w")]
    for h_key, w_key in pairs:
        h_val = ann.get(h_key)
        w_val = ann.get(w_key)
        if isinstance(h_val, (int, float)) and isinstance(w_val, (int, float)) and h_val > 0 and w_val > 0:
            return int(h_val), int(w_val)
    return default_hw


def _build_sample_entry(data_root: Path, item: str) -> SampleEntry:
    ann_path = _resolve_annotation_path(data_root, item)
    ann = _load_json(ann_path)
    img_path = _resolve_image_path(data_root, ann, item)
    file_path = _parse_file_path(ann, item)
    return SampleEntry(
        item=item,
        ann_path=ann_path,
        img_path=img_path,
        file_path=file_path,
    )


def _entry_to_dict(entry: SampleEntry) -> Dict[str, str]:
    return {
        "item": entry.item,
        "ann_path": str(entry.ann_path) if entry.ann_path is not None else "",
        "img_path": str(entry.img_path) if entry.img_path is not None else "",
        "file_path": entry.file_path,
    }


def _entry_from_dict(raw: Dict[str, str]) -> SampleEntry | None:
    item = str(raw.get("item", ""))
    if not item:
        return None
    ann_path_raw = str(raw.get("ann_path", ""))
    img_path_raw = str(raw.get("img_path", ""))
    ann_path = Path(ann_path_raw) if ann_path_raw else None
    img_path = Path(img_path_raw) if img_path_raw else None
    file_path = str(raw.get("file_path", "")) or _parse_file_path({}, item)
    return SampleEntry(
        item=item,
        ann_path=ann_path,
        img_path=img_path,
        file_path=file_path,
    )


def _preindex_cache_path(data_root: Path, list_path: str) -> Path:
    list_name = Path(list_path).stem or "list"
    cache_name = f"openlane_preindex_{list_name}.json"
    return data_root / ".cache" / cache_name


def _samples_fingerprint(samples: List[str]) -> str:
    payload = "\n".join(samples).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_preindex_cache(cache_path: Path, samples: List[str]) -> List[SampleEntry] | None:
    if not cache_path.exists():
        return None
    try:
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    expected_fp = _samples_fingerprint(samples)
    if isinstance(raw, dict):
        entries_raw = raw.get("entries")
        if not isinstance(entries_raw, list):
            return None
        if str(raw.get("samples_fingerprint", "")) != expected_fp:
            return None
    elif isinstance(raw, list):
        entries_raw = raw
    else:
        return None
    entries: List[SampleEntry] = []
    for item in entries_raw:
        if not isinstance(item, dict):
            return None
        entry = _entry_from_dict(item)
        if entry is None:
            return None
        entries.append(entry)
    if len(entries) != len(samples):
        return None
    if [entry.item for entry in entries] != list(samples):
        return None
    return entries


def _write_preindex_cache(cache_path: Path, entries: List[SampleEntry], samples: List[str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "samples_fingerprint": _samples_fingerprint(samples),
        "entries": [_entry_to_dict(entry) for entry in entries],
    }
    tmp_path = cache_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(cache_path)


def _build_sample_entries(
    data_root: Path,
    samples: List[str],
    list_path: str,
    preindex_cache: bool,
) -> List[SampleEntry]:
    cache_path = _preindex_cache_path(data_root, list_path)
    if preindex_cache:
        cached = _load_preindex_cache(cache_path, samples)
        if cached is not None:
            return cached
    entries = [_build_sample_entry(data_root, item) for item in samples]
    if preindex_cache:
        _write_preindex_cache(cache_path, entries, samples)
    return entries


def _as_tensor_matrix(
    value,
    shape: Tuple[int, int],
    default: np.ndarray,
    field_name: str,
    camera_param_policy: str,
) -> Tensor:
    if isinstance(value, list):
        arr = np.asarray(value, dtype=np.float32)
    elif isinstance(value, np.ndarray):
        arr = value.astype(np.float32)
    else:
        if camera_param_policy == "fallback":
            return torch.from_numpy(default.astype(np.float32))
        raise ValueError(f"相机参数缺失: {field_name}")
    if arr.shape != shape:
        if camera_param_policy == "fallback":
            return torch.from_numpy(default.astype(np.float32))
        raise ValueError(f"相机参数形状错误: {field_name}, 期望 {shape}, 实际 {arr.shape}")
    return torch.from_numpy(arr)


def _parse_cam_extrinsic(ann: Dict, camera_param_policy: str) -> Tensor:
    return _as_tensor_matrix(
        value=ann.get("extrinsic"),
        shape=(4, 4),
        default=DEFAULT_EXTRINSIC,
        field_name="extrinsic",
        camera_param_policy=camera_param_policy,
    )


def _parse_cam_intrinsic(ann: Dict, camera_param_policy: str) -> Tensor:
    return _as_tensor_matrix(
        value=ann.get("intrinsic"),
        shape=(3, 3),
        default=DEFAULT_INTRINSIC,
        field_name="intrinsic",
        camera_param_policy=camera_param_policy,
    )


class OpenLaneDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        list_path: str,
        image_size: Tuple[int, int],
        max_lanes: int,
        num_points: int,
        camera_param_policy: str = "strict",
        category_policy: str = "preserve_21",
        preindex_cache: bool = True,
        image_backend: str = "pil",
        enable_timing_probe: bool = False,
    ) -> None:
        if camera_param_policy not in {"strict", "fallback"}:
            raise ValueError("camera_param_policy 仅支持 strict/fallback")
        if category_policy not in {"preserve_21", "legacy_map_21_to_20"}:
            raise ValueError("category_policy 仅支持 preserve_21/legacy_map_21_to_20")
        if image_backend not in {"pil", "cv2"}:
            raise ValueError("image_backend 仅支持 pil/cv2")
        self.data_root = Path(data_root).resolve()
        self.samples = _read_list_file((self.data_root / list_path).resolve())
        self.sample_entries = _build_sample_entries(
            data_root=self.data_root,
            samples=self.samples,
            list_path=list_path,
            preindex_cache=bool(preindex_cache),
        )
        self.image_size = image_size
        self.max_lanes = max_lanes
        self.num_points = num_points
        self.camera_param_policy = camera_param_policy
        self.category_policy = category_policy
        self.image_backend = image_backend
        self.enable_timing_probe = bool(enable_timing_probe)

    def __len__(self) -> int:
        return len(self.sample_entries)

    def __getitem__(self, idx: int) -> Dict[str, Tensor | Dict[str, str]]:
        entry = self.sample_entries[idx]
        item = entry.item
        ann_path = entry.ann_path
        img_path = entry.img_path
        sample_timing = None
        if self.enable_timing_probe:
            start_t = time.perf_counter()
            ann_start_t = time.perf_counter()
            ann = _load_json(ann_path)
            ann_t = time.perf_counter() - ann_start_t

            image_start_t = time.perf_counter()
            image, image_hw = _load_image(img_path, self.image_size, self.image_backend)
            image_t = time.perf_counter() - image_start_t

            lanes_start_t = time.perf_counter()
            src_h, src_w = _parse_src_image_hw(ann, image_hw)
            gt_points, gt_vis, lane_valid, lane_category = _parse_lanes(
                ann=ann,
                max_lanes=self.max_lanes,
                num_points=self.num_points,
                category_policy=self.category_policy,
            )
            lanes_t = time.perf_counter() - lanes_start_t

            camera_start_t = time.perf_counter()
            cam_extrinsic = _parse_cam_extrinsic(ann, self.camera_param_policy)
            cam_intrinsic = _parse_cam_intrinsic(ann, self.camera_param_policy)
            camera_t = time.perf_counter() - camera_start_t
            total_t = time.perf_counter() - start_t
            sample_timing = torch.tensor(
                [ann_t, image_t, lanes_t, camera_t, total_t],
                dtype=torch.float32,
            )
        else:
            ann = _load_json(ann_path)
            image, image_hw = _load_image(img_path, self.image_size, self.image_backend)
            src_h, src_w = _parse_src_image_hw(ann, image_hw)
            gt_points, gt_vis, lane_valid, lane_category = _parse_lanes(
                ann=ann,
                max_lanes=self.max_lanes,
                num_points=self.num_points,
                category_policy=self.category_policy,
            )
            cam_extrinsic = _parse_cam_extrinsic(ann, self.camera_param_policy)
            cam_intrinsic = _parse_cam_intrinsic(ann, self.camera_param_policy)
        file_path = entry.file_path
        meta = {
            "sample_id": item,
            "image_path": str(img_path) if img_path is not None else "",
            "ann_path": str(ann_path) if ann_path is not None else "",
            "file_path": file_path,
        }
        output: Dict[str, Tensor | Dict[str, str]] = {
            "image": image,
            "gt_points": gt_points,
            "gt_vis": gt_vis,
            "lane_valid": lane_valid,
            "gt_category": lane_category,
            "cam_extrinsic": cam_extrinsic,
            "cam_intrinsic": cam_intrinsic,
            "src_img_hw": torch.tensor([float(src_h), float(src_w)], dtype=torch.float32),
            "meta": meta,
        }
        if sample_timing is not None:
            output["sample_timing"] = sample_timing
        return output


def openlane_collate(batch: List[Dict]) -> Dict[str, Tensor | List[Dict[str, str]]]:
    keys = ["image", "gt_points", "gt_vis", "lane_valid", "gt_category", "cam_extrinsic", "cam_intrinsic", "src_img_hw"]
    out = {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}
    if "sample_timing" in batch[0]:
        out["sample_timing"] = torch.stack([item["sample_timing"] for item in batch], dim=0)
    out["meta"] = [item["meta"] for item in batch]
    return out
