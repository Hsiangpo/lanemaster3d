# 论文复现规范

## 1. 环境记录

1. CUDA版本  
2. PyTorch版本  
3. Python版本  
4. GPU型号与数量  

## 2. 配置记录

每次实验必须保存：

1. 配置文件快照  
2. 随机种子  
3. 训练命令  
4. 评测命令  

## 3. 日志记录

1. 训练日志路径  
2. 评测日志路径  
3. 关键指标时间点  

## 4. 权重与输出

1. best checkpoint  
2. latest checkpoint  
3. 评测JSON结果  
4. 可视化样例  

## 5. 最小复现实验

```bash
bash scripts/train_dist.sh configs/openlane/r50_960x720.py experiments/exp_repro 2
bash scripts/eval_dist.sh configs/openlane/r50_960x720.py experiments/exp_repro/latest.pth experiments/eval_repro 2
```

## 6. 投稿前核查清单

1. 三个随机种子结果是否齐全  
2. 主表与消融表是否可追溯到日志  
3. 关键图表是否可由脚本重生成  

