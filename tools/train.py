from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lanemaster3d.command_builder import build_train_bundle
from lanemaster3d.config_loader import load_python_config, validate_config
from lanemaster3d.engine import train_model
from lanemaster3d.runtime import run_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LaneMaster3D 训练入口")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--work-dir", required=True, help="实验目录")
    parser.add_argument("--gpus", type=int, default=2, help="GPU数量")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令")
    parser.add_argument("--local-run", action="store_true", help="内部执行标记")
    parser.add_argument("--launcher", default="none", choices=["none", "ddp"], help="分布式启动器")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    if not args.local_run:
        bundle = build_train_bundle(args.config, args.work_dir, args.gpus)
        return run_bundle(bundle, dry_run=args.dry_run)
    config = load_python_config(args.config)
    config = validate_config(config)
    metric = train_model(config, args.work_dir, launcher=args.launcher)
    print(metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
