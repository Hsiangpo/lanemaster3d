# LaneMaster3D 远端执行手册

## 1. 连接与会话

```bash
ssh -p 2222 sust@38.127.121.195
cd /home/sust/zhou/lanemaster3d
tmux new -s lm3d
```

## 2. 基础检查

```bash
bash scripts/remote/check_env.sh
bash scripts/remote/check_openlane_data.sh .
```

## 3. 训练启动

```bash
bash scripts/train_dist.sh configs/openlane/r50_960x720.py experiments/lm3d_exp001 2
```

说明：`scripts/train_dist.sh`在多卡下会自动使用`torchrun --nproc_per_node=<GPU数>`启动标准DDP训练。

## 4. 评测启动

```bash
bash scripts/eval_dist.sh configs/openlane/r50_960x720.py experiments/lm3d_exp001/latest.pth experiments/lm3d_eval001 2
```

## 5. 断点续训

```bash
bash scripts/resume.sh configs/openlane/r50_960x720.py experiments/lm3d_exp001 2
```

## 6. 后台长训（tmux）

```bash
bash scripts/remote/start_train_tmux.sh lm3d_train configs/openlane/r50_960x720.py experiments/lm3d_exp001 2
tmux attach -t lm3d_train
```

## 7. 常用排障

1. GPU不可见：检查`nvidia-smi`与CUDA版本。  
2. 数据路径错误：检查`/home/sust/zhou/lanemaster3d/data/OpenLane`目录是否完整。  
3. 训练日志查看：检查`experiments/<exp>/logs/metrics.jsonl`与`logs/train_*.log`。  
4. 训练中断：确认`latest.pth`存在后执行`resume.sh`。  


