# 实验映射表（创新点 -> 实验方案）

## E1 时序多帧融合（对应 G01）

1. 改动：在当前 query token 前加入轻量时序融合（2-3帧）。
2. 对照：
   - Baseline 单帧
   - Baseline + 简单拼接
   - Baseline + 时序注意力融合
3. 指标：F_score、x/z close/far、吞吐、显存。

## E2 不确定性回归（对应 G02）

1. 改动：新增 x/z 的 log-variance 头，采用异方差损失。
2. 对照：
   - Baseline
   - Baseline + uncertainty loss
   - Baseline + uncertainty loss + confidence-aware NMS/阈值
3. 指标：F_score、误差方差、困难样本稳定性。

## E3 自适应稀疏锚点（对应 G03）

1. 改动：增加 anchor selection gate，按场景动态选择候选锚点。
2. 对照：
   - 全量锚点
   - 固定稀疏锚点
   - 自适应稀疏锚点
3. 指标：F_score、FPS、显存、参数量。

## E4 类别-几何一致性正则（对应 G06）

1. 改动：引入类别预测与曲率/可见性一致性损失。
2. 对照：
   - 仅分类损失
   - 分类 + 几何一致性
3. 指标：cate_acc、F_score、误检类别占比。

## E5 官方指标 best 路径回归测试（对应 G10）

1. 改动：补 trainer 单测，mock quick_f1 与 F_score 冲突场景。
2. 验收：best checkpoint 必须按 F_score 选择。

## 资源约束与执行顺序（2x3090）

1. 先跑 E5（低成本，先稳工程）。
2. 再跑 E2（最可能带来论文创新收益）。
3. 随后 E1（第二创新点，增强场景稳定性）。
4. 最后 E3 与 E4 做扩展消融。
