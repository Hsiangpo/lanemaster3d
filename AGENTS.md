# Repository Guidelines

## Project Structure & Module Organization
`lanemaster3d/` is the core package: `models/` (network heads and geometry), `models/losses/`, `data/` (OpenLane dataset + collate), `engine/` (train/eval loops), and `metrics/` (quick + official protocol).  
`configs/openlane/*.py` stores experiment configs.  
`tools/` provides runnable entrypoints (`train.py`, `eval.py`, `infer.py`).  
`scripts/` contains shell wrappers for local and remote execution.  
`tests/` contains pytest suites (`test_*.py`).  
`docs/` stores ops, planning, product, and research documents.

## Build, Test, and Development Commands
- `python tools/train.py --config configs/openlane/r50_960x720.py --work-dir experiments/demo --gpus 2 --dry-run`  
  Validate train launch arguments without starting training.
- `bash scripts/train_dist.sh configs/openlane/r50_960x720.py experiments/exp001 2`  
  Start distributed training.
- `python tools/eval.py --config <cfg> --ckpt <pth> --out <dir> --dry-run`  
  Validate eval command wiring.
- `bash scripts/eval_dist.sh configs/openlane/r50_960x720.py experiments/exp001/best.pth experiments/eval001 2`  
  Run distributed evaluation.
- `pytest -q` or `pytest -q tests/test_openlane_dataset.py`  
  Run full or targeted test suites.

## Remote Execution Rule (新增)
训练、评测、推理一律按远端环境执行，不以本地开发机结果作为最终口径。  
固定远端项目目录：`/home/sust/zhou/lanemaster3d`（详见 `docs/ops/server.md`）。  
开训前必须先执行：
- `bash scripts/remote/check_env.sh`
- `bash scripts/remote/check_openlane_data.sh .`
训练建议优先使用：
- `bash scripts/train_dist.sh <config> <work_dir> <gpus>`
- 或 `bash scripts/remote/start_train_tmux.sh <session> <config> <work_dir> <gpus>`
若仅在本地执行命令，需明确标注该结果仅用于静态检查或命令拼装验证，不代表远端训练结论。

## Coding Style & Naming Conventions
Use Python with 4-space indentation and UTF-8 encoding.  
Use `snake_case` for functions/files/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.  
Add type hints for new public functions.  
Keep functions under 80 lines and keep files near/under 500 lines by splitting modules when needed.  
Code comments should be Chinese and explain intent, not obvious syntax.

## Testing Guidelines
Framework: `pytest`.  
Test files must follow `tests/test_*.py`, and test functions must start with `test_`.  
For bug fixes, add a failing regression test first, then implement the fix.  
When changing launch scripts or distributed paths, add command-level tests (for example, DDP launcher assertions).

## Commit & Pull Request Guidelines
Follow the observed commit style: `type: summary` (example: `fix: 对齐推理配置并修复评测DDP链路`).  
Keep one logical change per commit.  
PRs should include: purpose, key module changes, exact test commands and results, and config/data assumptions.  
For training or evaluation changes, include artifact paths such as `experiments/<exp>/metrics.json` and `<eval_out>/eval_metrics.json`.

## Ops & Cleanup Notes
Before remote training/deployment, read `docs/ops/server.md`.  
Do not leave `.tmp_*` files in the repo or remote workspace; move reusable scripts into `scripts/`.

## Docs Overview (补充)
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
