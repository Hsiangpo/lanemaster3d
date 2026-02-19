from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not torch.is_tensor(labels):
        raise TypeError(f"labels类型错误: {type(labels)}")
    if labels.dtype != torch.int64:
        raise ValueError(f"labels必须为int64: {labels.dtype}")
    if num_classes < 1:
        raise ValueError(f"num_classes必须大于0: {num_classes}")
    shape = labels.shape
    out = torch.zeros(shape[0], num_classes, *shape[1:], device=device, dtype=dtype)
    return out.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def focal_loss(
    inputs: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = "none",
    eps: float = 1e-8,
) -> torch.Tensor:
    if not torch.is_tensor(inputs):
        raise TypeError(f"inputs类型错误: {type(inputs)}")
    if len(inputs.shape) < 2:
        raise ValueError(f"inputs维度错误: {inputs.shape}")
    if inputs.size(0) != target.size(0):
        raise ValueError(f"batch不匹配: {inputs.size(0)} vs {target.size(0)}")
    out_size = (inputs.size(0),) + inputs.size()[2:]
    if target.size()[1:] != inputs.size()[2:]:
        raise ValueError(f"target尺寸错误: 期望{out_size}, 实际{target.size()}")
    if inputs.device != target.device:
        raise ValueError(f"device不一致: {inputs.device} vs {target.device}")
    input_soft = F.softmax(inputs, dim=1) + eps
    target_one_hot = one_hot(target, inputs.shape[1], inputs.device, inputs.dtype)
    weight = torch.pow(1.0 - input_soft, gamma)
    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)
    if reduction == "none":
        return loss_tmp
    if reduction == "mean":
        return torch.mean(loss_tmp)
    if reduction == "sum":
        return torch.sum(loss_tmp)
    raise NotImplementedError(f"不支持的reduction: {reduction}")


class FocalLoss(nn.Module):
    """多类Focal Loss。"""

    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = "none") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(inputs, target, self.alpha, self.gamma, self.reduction, self.eps)

