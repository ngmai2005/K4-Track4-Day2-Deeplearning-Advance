"""timm backbones plus optimizer groups and model complexity accounting."""
from __future__ import annotations
import torch
from torch import nn

SUGGESTED_BACKBONES = {"resnet50":"resnet50", "resnext50":"resnext50_32x4d", "convnext_tiny":"convnext_tiny", "deit_small":"deit_small_patch16_224", "swin_tiny":"swin_tiny_patch4_window7_224", "efficientnet_b0":"efficientnet_b0", "mobilenetv3":"mobilenetv3_large_100"}

def build_model(name, pretrained=True, num_classes=9, drop_rate=0., init="finetune"):
    import timm
    if init not in {"scratch", "frozen", "finetune"}: raise ValueError(f"Unknown init: {init}")
    model = timm.create_model(name, pretrained=pretrained and init != "scratch", num_classes=num_classes, drop_rate=drop_rate)
    model.pretrained_tag = getattr(model, "pretrained_cfg", {}).get("tag", "scratch" if init == "scratch" else "unknown")
    if init == "frozen": freeze_backbone(model)
    return model

def freeze_backbone(model):
    head = model.get_classifier(); head_ids = {id(p) for p in head.parameters()}
    for p in model.parameters(): p.requires_grad = id(p) in head_ids
    model.backbone_frozen = True

def param_groups(model, lr_backbone, lr_head, weight_decay):
    head_ids = {id(p) for p in model.get_classifier().parameters()}; back_decay=[]; back_no_decay=[]; head=[]
    for p in model.parameters():
        if not p.requires_grad: continue
        if id(p) in head_ids: head.append(p)
        elif p.ndim <= 1: back_no_decay.append(p)
        else: back_decay.append(p)
    return [{"params":back_decay,"lr":lr_backbone,"weight_decay":weight_decay}, {"params":back_no_decay,"lr":lr_backbone,"weight_decay":0.}, {"params":head,"lr":lr_head,"weight_decay":weight_decay}]

def count_params(model): return sum(p.numel() for p in model.parameters()) / 1e6
def count_gmacs(model, img_size=224):
    try:
        from thop import profile
        device = next(model.parameters()).device
        example = torch.zeros((1, 3, img_size, img_size), device=device)
        macs, _ = profile(model.eval(), inputs=(example,), verbose=False)
        return macs / 1e9
    except Exception: return float("nan")
