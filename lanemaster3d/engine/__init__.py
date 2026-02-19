"""训练与评测引擎。"""

from .trainer import train_model
from .evaluator import evaluate_model

__all__ = ["train_model", "evaluate_model"]

