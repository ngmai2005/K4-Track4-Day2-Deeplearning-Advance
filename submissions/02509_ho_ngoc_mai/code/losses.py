"""Losses and batch mixing used by all DeepWeeds experiments."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    weight = kw.get("weight")
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1), weight=weight)
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha", weight))
    if kind == "ce_weighted":
        if weight is None:
            raise ValueError("ce_weighted requires a train-set class-weight tensor")
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"Unknown loss: {kind}")


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1, weight=None):
        super().__init__()
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing, self.weight = smoothing, weight

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, weight=self.weight,
                               label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target[:, None]).squeeze(1)
        pt = logpt.exp()
        loss = -(1 - pt).pow(self.gamma) * logpt
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    counts = torch.as_tensor(counts, dtype=torch.float32)
    if (counts <= 0).any():
        raise ValueError("All classes must occur in the training split")
    if beta == 0:
        w = counts.reciprocal()
    else:
        if not 0 < beta < 1:
            raise ValueError("beta must be 0 or in (0, 1)")
        w = (1 - beta) / (1 - beta ** counts)
    return w / w.mean()


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    if alpha <= 0:
        return x, (y, y, 1.0)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        return x.mul(lam).add(x[perm], alpha=1 - lam), (y, y[perm], lam)
    if mode != "cutmix":
        raise ValueError(f"Unknown mix mode: {mode}")
    _, _, h, w = x.shape
    cut = np.sqrt(1 - lam)
    cw, ch = int(w * cut), int(h * cut)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
    y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
    mixed = x.clone(); mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    return mixed, (y, y[perm], float(lam))


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
