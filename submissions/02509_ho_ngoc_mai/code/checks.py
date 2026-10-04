"""Small correctness checks required before expensive DeepWeeds experiments."""
from __future__ import annotations
import torch
from torch import nn
from losses import FocalLoss, mix_batch
from inference import fuse_conv_bn


def check_focal_gamma_zero() -> float:
    torch.manual_seed(0)
    logits = torch.randn(11, 9); labels = torch.randint(0, 9, (11,))
    delta = float((FocalLoss(gamma=0)(logits, labels) - nn.CrossEntropyLoss()(logits, labels)).abs())
    assert delta < 1e-6, f"focal(gamma=0) differs from CE by {delta}"
    return delta


def check_cutmix_area() -> float:
    torch.manual_seed(0)
    x = torch.rand(8, 3, 32, 32); y = torch.arange(8)
    mixed, (_, _, lam) = mix_batch(x, y, alpha=1.0, mode="cutmix")
    assert mixed.shape == x.shape and 0 <= lam <= 1
    return float(lam)


def check_fuse_conv_bn() -> float:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU()).eval()
    x = torch.randn(3, 3, 16, 16)
    fused = fuse_conv_bn(model)
    delta = float((model(x) - fused(x)).abs().max().detach())
    assert delta < 1e-5, f"Conv-BN fusion error: {delta}"
    return delta


def main() -> None:
    print({"focal_gamma0_delta": check_focal_gamma_zero(), "cutmix_lambda": check_cutmix_area(), "fuse_max_abs_delta": check_fuse_conv_bn()})


if __name__ == "__main__":
    main()
