from __future__ import annotations


def average_metric_list(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "x_error": 0.0, "z_error": 0.0}
    keys = sorted(metrics[0].keys())
    count = float(len(metrics))
    return {k: float(sum(float(m.get(k, 0.0)) for m in metrics) / count) for k in keys}


def should_run_official_eval(epoch: int, total_epochs: int, interval: int) -> bool:
    if interval <= 0:
        return False
    return ((epoch + 1) % interval == 0) or (epoch + 1 == total_epochs)
