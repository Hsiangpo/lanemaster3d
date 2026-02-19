# 协作说明

## 文档总览

`docs` 目录已按用途分组，开发时按下列目录查阅：

- `docs/ops`：运维与服务器接入文档。
  - `docs/ops/server.md`：服务器地址、固定SSH入口、账号与路径、连接状态、排障记录。
  - `docs/ops/Runbook.MD`：日常执行命令、训练/监控/恢复操作手册。
- `docs/product`：需求定义文档。
  - `docs/product/PRD.MD`：产品与功能需求基线，功能实现和验收以此为准。
- `docs/plans`：执行计划与阶段计划快照。
  - `docs/plans/Plan.MD`：主计划文档。
  - `docs/plans/20260219_1808_gpu-util-plan.md`：GPU利用率优化阶段计划快照。
- `docs/experiments`：实验规范与实验协同文档。
  - `docs/experiments/ExperimentSpec.MD`：实验标准、配置规范、记录约定。
  - `docs/experiments/Paper_Shared_Experiments.MD`：论文相关共享实验设计与对齐项。
- `docs/papers`：论文写作与投稿材料。
  - `docs/papers/Paper_Conf_Outline.MD`：会议论文大纲。
  - `docs/papers/Paper_Innovation_Survey.MD`：创新点调研摘要。
  - `docs/papers/Paper_Journal_Extension.MD`：期刊扩展规划。
  - `docs/papers/Paper_Reproducibility.MD`：可复现性与实验复现说明。
- `docs/research`：深度文献调研与证据链。
  - `docs/research/paper_registry.md`：文献清单与状态。
  - `docs/research/claim_traceability.md`：结论与证据追溯。
  - `docs/research/evidence_cards/`：主题证据卡片。
  - `docs/research/bibliography.bib`：BibTeX 引用库。

## 使用要求

- 处理远端训练、部署、数据同步前，先读 `docs/ops/server.md`。
- 做实验配置变更前，先读 `docs/experiments/ExperimentSpec.MD`。
- 做方案或排期变更时，更新 `docs/plans` 对应文档。

## 临时文件清理规则（强制）

- 以 `.tmp_` 开头的临时脚本、临时配置、临时日志，使用完成后必须立即删除。
- 本地仓库根目录禁止残留 `.tmp_*` 文件。
- 远端项目目录（如 `/home/sust/zhou/lanemaster3d`）也禁止残留 `.tmp_*` 文件。
- 需要长期复用的脚本不得放在 `.tmp_*`，必须迁移到 `scripts/` 并使用正式命名。
- 每次任务结束前必须执行一次临时文件检查，确认无 `.tmp_*` 残留。
