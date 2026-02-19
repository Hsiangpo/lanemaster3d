# 文献总表（已登记）

> 说明：本表仅收录一手来源。所有在研究文档中出现的引用均在此登记。

| ID | Year | 类型 | 标题 | 一手来源 | 官方代码 | 对 LM3D 的直接价值 |
|---|---:|---|---|---|---|---|
| P001 | 2017 | 基础方法 | Spatial As Deep: Spatial CNN for Traffic Scene Understanding | https://arxiv.org/abs/1712.06080 | https://github.com/XingangPan/SCNN | 经典空间传播建模基线 |
| P002 | 2017 | 基础损失 | Focal Loss for Dense Object Detection | https://arxiv.org/abs/1708.02002 | https://github.com/facebookresearch/Detectron | 分类困难样本加权理论依据 |
| P003 | 2017 | 基础方法 | Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics | https://arxiv.org/abs/1705.07115 | - | 不确定性加权损失理论依据 |
| P004 | 2018 | 3D车道 | 3D-LaneNet: End-to-End 3D Multiple Lane Detection | https://arxiv.org/abs/1811.10203 | https://github.com/yuliangguo/3D_Lane_Synthetic_Dataset | 单目3D车道早期端到端基线 |
| P005 | 2019 | 2D车道 | FastDraw: Addressing the Long Tail of Lane Detection by Adapting a Sequential Prediction Network | https://arxiv.org/abs/1905.04354 | - | 车道长尾分布建模参考 |
| P006 | 2020 | 3D车道 | Gen-LaneNet: A Generalized and Scalable Approach for 3D Lane Detection | https://arxiv.org/abs/2003.10656 | https://github.com/yuliangguo/Gen-LaneNet | 泛化型3D车道建模思路 |
| P007 | 2020 | 3D车道 | Semi-Local 3D Lane Detection and Uncertainty Estimation | https://arxiv.org/abs/2003.05257 | https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection | 3D车道+不确定性联合建模 |
| P008 | 2020 | 2D车道 | Ultra Fast Structure-aware Deep Lane Detection | https://arxiv.org/abs/2004.11757 | https://github.com/cfzd/Ultra-Fast-Lane-Detection | 高效训练与部署基线 |
| P009 | 2020 | 2D车道 | PolyLaneNet: Lane Estimation via Deep Polynomial Regression | https://arxiv.org/abs/2004.10924 | https://github.com/lucastabelini/PolyLaneNet | 曲线参数化建模参考 |
| P010 | 2020 | 基础Transformer | End-to-End Object Detection with Transformers (DETR) | https://arxiv.org/abs/2005.12872 | https://github.com/facebookresearch/detr | query-based检测框架基础 |
| P011 | 2020 | 基础Transformer | Deformable DETR: Deformable Transformers for End-to-End Object Detection | https://arxiv.org/abs/2010.04159 | https://github.com/fundamentalvision/Deformable-DETR | 可形变注意力与采样机制参考 |
| P012 | 2020 | 2D车道 | Keep your Eyes on the Lane: Real-time Attention-guided Lane Detection | https://arxiv.org/abs/2010.12035 | https://github.com/lucastabelini/LaneATT | anchor/query式车道检测参考 |
| P013 | 2021 | 2D车道 | CondLaneNet: A Top-to-down Lane Detection Framework Based on Conditional Convolutions | https://arxiv.org/abs/2105.05003 | https://github.com/aliyun/conditional-lane-detection | 条件卷积车道建模参考 |
| P014 | 2021 | 2D车道 | Robust Lane Detection from Continuous Driving Scenes Using Deep Neural Networks | https://arxiv.org/abs/2103.12040 | https://github.com/sel118/LaneAF | 连续场景鲁棒训练参考 |
| P015 | 2022 | 3D车道+数据集 | PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark | https://arxiv.org/abs/2203.11089 | https://github.com/OpenDriveLab/PersFormer_3DLane | OpenLane任务定义与主流对照 |
| P016 | 2022 | 2D车道 | CLRNet: Cross Layer Refinement Network for Lane Detection | https://arxiv.org/abs/2203.10350 | https://github.com/Turoad/CLRNet | 多层特征精化策略参考 |
| P017 | 2022 | 数据集 | ONCE-3DLanes: Building Monocular 3D Lane Detection | https://arxiv.org/abs/2205.00301 | https://github.com/Once-3DLanes/once_3dlanes_benchmark | 跨数据集泛化评测补充 |
| P018 | 2022 | 3D车道 | Reconstruct from BEV: A 3D Lane Detection Approach based on Geometry Structure Prior | https://arxiv.org/abs/2206.10098 | https://github.com/gigo-team/reconstruct-from-bev | 几何先验融合方案参考 |
| P019 | 2022 | 多模态3D车道 | M$^2$-3DLaneNet: Exploring Multi-Modal 3D Lane Detection | https://arxiv.org/abs/2209.05996 | - | 多模态上限对照 |
| P020 | 2022 | 3D车道 | WS-3D-Lane: Weakly Supervised 3D Lane Detection With 2D Lane Labels | https://arxiv.org/abs/2209.11523 | - | 弱监督标注成本优化参考 |
| P021 | 2022 | 3D车道 | BEV-LaneDet: A Simple and Effective 3D Lane Detection Baseline | https://arxiv.org/abs/2210.06006 | https://github.com/Sephirex-X/BEV-LaneDet | BEV基线与效率对照 |
| P022 | 2023 | 3D车道 | Anchor3DLane: Learning to Regress 3D Anchors for Monocular 3D Lane Detection | https://arxiv.org/abs/2301.02371 | https://github.com/tusen-ai/Anchor3DLane | 直接竞品与结构参考主对象 |
| P023 | 2023 | 拓扑推理 | Graph-based Topology Reasoning for Driving Scenes | https://arxiv.org/abs/2304.05277 | https://github.com/jbji/RepVF | 拓扑图建模参考 |
| P024 | 2023 | 数据集+任务 | OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping | https://arxiv.org/abs/2304.10440 | https://github.com/OpenDriveLab/OpenLane-V2 | 论文延展到拓扑任务的依据 |
| P025 | 2023 | 3D车道 | An Efficient Transformer for Simultaneous Learning of BEV and Lane Representations in 3D Lane Detection | https://arxiv.org/abs/2306.04927 | - | 轻量Transformer方向参考 |
| P026 | 2023 | 拓扑推理 | TopoMask: Instance-Mask-Based Formulation for the Road Topology Problem | https://arxiv.org/abs/2306.05419 | https://github.com/wudongming97/TopoMLP | 实例mask拓扑建模参考 |
| P027 | 2023 | 3D车道 | GroupLane: End-to-End 3D Lane Detection with Channel-wise Grouping | https://arxiv.org/abs/2307.09472 | - | 通道分组机制参考 |
| P028 | 2023 | 在线建图 | Online Monocular Lane Mapping Using Catmull-Rom Spline | https://arxiv.org/abs/2307.11653 | - | 曲线建图表达参考 |
| P029 | 2023 | 3D车道 | LATR: 3D Lane Detection from Monocular Images with Transformer | https://arxiv.org/abs/2308.04583 | https://github.com/JMoonr/LATR | Transformer 3D车道强基线 |
| P030 | 2023 | 拓扑推理 | TopoMLP: A Simple yet Strong Pipeline for Driving Topology Reasoning | https://arxiv.org/abs/2310.06753 | https://github.com/wudongming97/TopoMLP | 拓扑推理pipeline参考 |
| P031 | 2023 | 车道分段+拓扑 | LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving | https://arxiv.org/abs/2312.16108 | https://github.com/OpenDriveLab/LaneSegNet | lane segment表达参考 |
| P032 | 2024 | 3D车道 | 3D Lane Detection from Front or Surround-View using Joint-Modeling & Matching | https://arxiv.org/abs/2401.08036 | - | 前视/环视联合建模参考 |
| P033 | 2024 | 综述 | Monocular 3D Lane Detection for Autonomous Driving: Recent Achievements, Challenges, and Outlooks | https://arxiv.org/abs/2404.06860 | - | 现状梳理与空白点来源 |
| P034 | 2024 | 拓扑推理 | TopoLogic: An Interpretable Pipeline for Lane Topology Reasoning on Driving Scenes | https://arxiv.org/abs/2405.14747 | - | 可解释拓扑推理路径参考 |
| P035 | 2024 | 3D车道+拓扑 | Enhancing 3D Lane Detection and Topology Reasoning with 2D Lane Priors | https://arxiv.org/abs/2406.03105 | - | 2D先验辅助3D/拓扑参考 |
| P036 | 2024 | 3D车道 | LaneCPP: Continuous 3D Lane Detection using Physical Priors | https://arxiv.org/abs/2406.08381 | - | 物理先验连续表示参考 |
| P037 | 2024 | 3D车道 | HeightLane: BEV Heightmap Guided 3D Lane Detection | https://arxiv.org/abs/2408.08270 | - | 高度图引导机制参考 |
| P038 | 2024 | 3D车道 | Anchor3DLane++: 3D Lane Detection via Sample-Adaptive Sparse 3D Anchor Regression | https://arxiv.org/abs/2412.16889 | https://github.com/tusen-ai/Anchor3DLane | 稀疏锚点创新直接参考 |
| P039 | 2025 | 3D车道 | Depth3DLane: Monocular 3D Lane Detection via Depth Prior Distillation | https://arxiv.org/abs/2504.18325 | - | 深度先验蒸馏方向参考 |
| P040 | 2025 | 时序3D车道 | Breaking Down Monocular Ambiguity: Exploiting Temporal Evolution for 3D Lane Detection | https://arxiv.org/abs/2504.20525 | - | 时序信息增强参考 |
| P041 | 2025 | 3D车道 | DB3D-L: Depth-aware BEV Feature Transformation for Accurate 3D Lane Detection | https://arxiv.org/abs/2505.13266 | - | 深度感知BEV变换参考 |
| P042 | 2025 | 时序拓扑 | TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving | https://arxiv.org/abs/2507.00709 | - | 时序拓扑建模参考 |
| P043 | 2025 | 3D车道 | SC-Lane: Slope-aware and Consistent Road Height Estimation Framework for 3D Lane Detection | https://arxiv.org/abs/2508.10411 | - | 坡度一致性建模参考 |
| P044 | 2025 | 3D车道+不确定性 | Monocular 3D Lane Detection via Structure Uncertainty-Aware Network with Curve-Point Queries | https://arxiv.org/abs/2511.13055 | - | 不确定性与曲线点查询参考 |

## 统计

1. 已登记文献：44 篇。
2. 3D车道/拓扑主线：31 篇。
3. 基础方法与损失：13 篇。
4. 仅一手来源：100%。
