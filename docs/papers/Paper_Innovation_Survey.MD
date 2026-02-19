# LaneMaster3D 论文调研与创新映射（2026-02-18）

## 1. 目标

本文档用于把“论文调研 -> 可复现实现 -> 训练评测策略”串起来，避免停留在概念层。

## 2. 关键论文（主来源）

1. PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark  
   来源：CVPR 2022 OpenAccess  
   链接：https://openaccess.thecvf.com/content/CVPR2022/html/Chen_PersFormer_3D_Lane_Detection_via_Perspective_Transformer_and_the_OpenLane_CVPR_2022_paper.html

2. LATR: 3D Lane Detection from Monocular Images with Transformer  
   来源：ICCV 2023 OpenAccess  
   链接：https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_LATR_3D_Lane_Detection_from_Monocular_Images_with_Transformer_ICCV_2023_paper.html

3. Anchor3DLane++: Improving 3D Lane Detection with Local Maxima Prior and Sample-adaptive Sparse Anchors  
   来源：arXiv / PAMI方向  
   链接：https://arxiv.org/abs/2401.16000

4. MonoUnc: Learning Uncertainty-Aware Monocular 3D Lane Detection  
   来源：arXiv 2025  
   链接：https://arxiv.org/abs/2511.13055

5. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics  
   来源：CVPR 2018  
   链接：https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html

## 3. 已落地创新（当前代码）

### 创新A：Depth-Aware Masked Token Aggregation（新增）

核心思想：参考Transformer类方法的“点位重要性建模”，在锚点采样特征聚合阶段，不再均值池化，而是使用`y深度 + 有效采样mask`生成自适应权重进行聚合，抑制越界采样噪声。

落地位置：

1. `lanemaster3d/models/lane_master_net.py` 中 `DepthAwareTokenAggregator`
2. `lanemaster3d/models/lane_master_net.py` 中 `_tokenize` 已改为mask加权聚合

收益预期：

1. 减少远距/越界点对token的污染
2. 提升复杂场景下稳定性，尤其是遮挡与大曲率区域

### 创新B：Curvature-Visibility Geometry Gating（已有并保留）

核心思想：使用曲率、可见性等几何先验门控query token，增强几何一致性。

落地位置：

1. `lanemaster3d/models/lane_master_net.py` 中 `GeometryAwareRefiner`

## 4. 关键补齐（对齐Anchor级别能力）

1. 多类别建模已打通（不再二分类）：`num_category=21`
2. 评测导出类别不再固定1，改为从分类logits解析类别
3. best checkpoint 选择可按官方OpenLane `F_score` 驱动

对应落地文件：

1. `configs/openlane/r50_960x720.py`
2. `lanemaster3d/data/openlane_dataset.py`
3. `lanemaster3d/engine/trainer.py`
4. `lanemaster3d/engine/evaluator.py`

## 5. 下一阶段论文级增强（已规划）

### 方向1：不确定性回归分支（参考MonoUnc + Kendall 2018）

计划：

1. 为x/z回归增加`log_var`头
2. 使用异方差不确定性加权回归损失
3. 在评测中增加按不确定性筛选的高置信提交子集

### 方向2：样本自适应稀疏锚点（参考Anchor3DLane++）

计划：

1. 动态稀疏化锚点候选集
2. 按场景难度与曲率分布调节anchor密度
3. 在不降精度前提下降低推理开销

## 6. 实验执行建议

1. 先跑当前版本，拿到官方协议基线
2. 单独开分支加入“不确定性分支”，做消融
3. 最后叠加“样本自适应稀疏锚点”，形成论文主实验
