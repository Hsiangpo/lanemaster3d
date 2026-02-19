# LaneMaster3D

独立的`3D车道线检测`项目，训练、评测、推理全部在本仓库内完成，不依赖外部代码目录。

## 当前特性

1. 本地配置驱动训练：`configs/openlane/r50_960x720.py`
2. 重型模型：`ResNet-FPN + Anchor风格检测头`
3. 创新模块：`DAGP(动态锚点) + GPC(几何先验一致性) + GCA`
4. 标准多卡：`torchrun + DDP + AMP`
5. 终端化流程：`tools/*.py` + `scripts/*.sh`

## 快速开始

```bash
# 训练命令预览
python tools/train.py --config configs/openlane/r50_960x720.py --work-dir experiments/demo --gpus 2 --dry-run

# 评测命令预览
python tools/eval.py --config configs/openlane/r50_960x720.py --ckpt experiments/demo/best.pth --out experiments/demo_eval --gpus 2 --dry-run
```

## 一键脚本

```bash
bash scripts/train_dist.sh configs/openlane/r50_960x720.py experiments/exp001 2
bash scripts/eval_dist.sh configs/openlane/r50_960x720.py experiments/exp001/best.pth experiments/eval001 2
```

## 训练日志与产物

1. 训练结构化日志：`experiments/<exp>/logs/metrics.jsonl`
2. 验证最优指标：`experiments/<exp>/metrics.json`
3. 模型权重：`experiments/<exp>/latest.pth`、`experiments/<exp>/best.pth`

## 远端终端脚本

```bash
# 环境检查
bash scripts/remote/check_env.sh

# 数据检查（默认检查 ./data/OpenLane）
bash scripts/remote/check_openlane_data.sh .

# tmux后台启动训练
bash scripts/remote/start_train_tmux.sh lm3d_train configs/openlane/r50_960x720.py experiments/lm3d_exp001 2
```

## 目录说明

- `lanemaster3d/`: 核心Python包
- `configs/`: 配置文件
- `tools/`: 训练/评测/推理入口
- `scripts/`: 终端批处理脚本
- `docs/`: 规划与论文资产
- `tests/`: 单元测试
