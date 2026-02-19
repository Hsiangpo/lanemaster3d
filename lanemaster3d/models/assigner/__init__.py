"""匹配分配器。"""

from .topk_assigner import TopKLaneAssigner


def build_assigner(cfg: dict | None):
    if cfg is None:
        return TopKLaneAssigner()
    cfg = cfg.copy()
    assigner_type = cfg.pop("type", "TopKLaneAssigner")
    if assigner_type not in {"TopKLaneAssigner", "TopkAssigner"}:
        raise ValueError(f"不支持的分配器类型: {assigner_type}")
    return TopKLaneAssigner(**cfg)


__all__ = ["TopKLaneAssigner", "build_assigner"]

