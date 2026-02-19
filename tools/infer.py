from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lanemaster3d.config_loader import load_python_config, validate_config
from lanemaster3d.data import OpenLaneDataset, openlane_collate
from lanemaster3d.engine.common import build_project_matrix, to_device
from lanemaster3d.models import LaneMaster3DNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LaneMaster3D 推理入口")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--ckpt", required=True, help="模型权重")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--gpus", type=int, default=1, help="GPU数量")
    return parser.parse_args()


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


def _run_inference_batches(
    model,
    loader: torch.utils.data.DataLoader | list[dict[str, torch.Tensor]],
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    pred_points_list: list[torch.Tensor] = []
    pred_scores_list: list[torch.Tensor] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = to_device(raw_batch, device) if device is not None else raw_batch
            project_matrix = None
            if isinstance(batch, dict) and "cam_extrinsic" in batch and "cam_intrinsic" in batch:
                project_matrix = build_project_matrix(batch)
            output = model(batch["image"], project_matrix=project_matrix)
            pred_points_list.append(output["pred_points"].detach().cpu())
            pred_scores_list.append(output["pred_scores"].detach().cpu())
    if not pred_points_list:
        return {
            "pred_points": torch.empty(0, 0, 0, 0, dtype=torch.float32),
            "pred_scores": torch.empty(0, 0, dtype=torch.float32),
        }
    return {
        "pred_points": torch.cat(pred_points_list, dim=0),
        "pred_scores": torch.cat(pred_scores_list, dim=0),
    }


def main() -> int:
    args = parse_args()
    config = load_python_config(args.config)
    config = validate_config(config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(config, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    data_cfg = config["data"]
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
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=openlane_collate)
    model.eval()
    pred = _run_inference_batches(model, loader, device=device)
    torch.save(
        pred,
        out_dir / "inference_output.pt",
    )
    print({"output": str(out_dir / "inference_output.pt")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
