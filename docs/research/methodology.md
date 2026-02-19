# 深度调研方法学（LaneMaster3D）

## 1. 目标与范围

本轮调研目标：构建单目3D车道检测方向的可追溯证据库，用于支持论文创新点论证与工程实现。

范围：

1. 单目3D车道检测方法与评测（OpenLane/ONCE-3DLanes/OpenLane-V2）。
2. 与主线直接相关的时序建模、拓扑推理、不确定性建模与关键基础模块。
3. 仅纳入一手来源（论文原文页 + 官方代码仓库）。

## 2. 检索协议

执行日期：2026-02-18

数据源：

1. arXiv 原文与 API（`export.arxiv.org/api/query`）。
2. CVF OpenAccess（CVPR/ICCV/ECCV 论文原文页）。
3. 官方代码仓库（作者/机构维护）。

检索查询（核心）：

1. `all:"monocular 3D lane detection"`
2. `all:"lane detection" AND all:"uncertainty"`
3. `all:"OpenLane" OR all:"ONCE-3DLanes"`

纳入标准：

1. 与3D车道检测主链路或强相关技术模块直接相关。
2. 提供可追溯的一手原文链接。
3. 对本项目创新或实现决策有实际参考价值。

排除标准：

1. 无法定位原文或仅有二手解读。
2. 与车道检测/拓扑推理主线无关。
3. 重复条目（同文不同版本仅保留主记录）。

## 3. 证据抽取模板

每篇文献至少记录：

1. 题目、年份、来源、主链接。
2. 核心贡献（结构/损失/训练/评测）。
3. 与 LaneMaster3D 的可迁移结论。
4. 风险与复现代价（依赖、训练资源、协议差异）。

## 4. 质量控制

1. 最少文献数：30（本次交付超过该下限）。
2. 所有被引用文献必须出现在：
   - `paper_registry.md`
   - `bibliography.bib`
3. `claim_traceability.md` 中每条结论至少对应1篇文献编号。
4. 对关键结论优先采用“方法论文 + 官方代码/基准文档”双证据。

## 5. 本次调研产出索引

1. `docs/research/paper_registry.md`
2. `docs/research/bibliography.bib`
3. `docs/research/claim_traceability.md`
4. `docs/research/innovation_gap_matrix.md`
5. `docs/research/experiment_mapping.md`
6. `docs/research/evidence_cards/`
