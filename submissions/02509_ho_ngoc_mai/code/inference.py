"""Evaluation-safe TTA, calibration, ensembling and Conv-BN fusion."""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def predict_logits(model, loader, device, view=None):
    model.eval(); names=[]; labels=[]; out=[]; view = view or (lambda x:x)
    with torch.inference_mode():
        for x,y,n in loader:
            x = view(x.to(device, non_blocking=True)); out.append(model(x).float().cpu()); labels.extend(y.tolist()); names.extend(n)
    return names, np.asarray(labels), torch.cat(out).numpy()
def view_identity(x): return x
def view_hflip(x): return torch.flip(x, dims=(-1,))
def views_multicrop(x, crop):
    h,w=x.shape[-2:];
    if crop > min(h,w): raise ValueError("crop exceeds image size")
    ys=(0,h-crop,(h-crop)//2); xs=(0,w-crop,(w-crop)//2)
    crops=[x[:,:,y:y+crop,z:z+crop] for y,z in ((ys[0],xs[0]),(ys[0],xs[1]),(ys[1],xs[0]),(ys[1],xs[1]),(ys[2],xs[2]))]
    return crops
def views_multiscale(x, sizes): return [F.interpolate(x, size=(s,s), mode="bilinear", align_corners=False) for s in sizes]
def aggregate_views(logits_per_view, space="prob"):
    a=np.stack(logits_per_view)
    if space=="prob": return torch.softmax(torch.as_tensor(a),-1).mean(0).numpy()
    if space=="logit": return torch.softmax(torch.as_tensor(a.mean(0)),-1).numpy()
    raise ValueError("space must be prob or logit")
def ensemble_probs(list_of_probs):
    p=np.asarray(list_of_probs, dtype=np.float64)
    if p.ndim != 3: raise ValueError("Expected M x N x C probabilities")
    result=p.mean(0); return result/result.sum(1,keepdims=True)
def fit_temperature(val_logits, val_labels):
    logits=torch.tensor(val_logits,dtype=torch.float32); labels=torch.tensor(val_labels,dtype=torch.long); log_t=torch.zeros((),requires_grad=True); opt=torch.optim.LBFGS([log_t],lr=.1,max_iter=50,line_search_fn="strong_wolfe")
    def closure(): opt.zero_grad(); loss=F.cross_entropy(logits/log_t.exp().clamp(.05,20),labels); loss.backward(); return loss
    opt.step(closure); return float(log_t.exp().clamp(.05,20).detach())
def apply_temperature(logits,T): return torch.softmax(torch.as_tensor(logits)/float(T),-1).numpy()
def fuse_conv_bn(model):
    model=copy.deepcopy(model).eval()
    for parent in model.modules():
        children=list(parent.named_children())
        for (n1,m1),(n2,m2) in zip(children,children[1:]):
            if isinstance(m1,nn.Conv2d) and isinstance(m2,nn.BatchNorm2d):
                fused=torch.nn.utils.fuse_conv_bn_eval(m1,m2); setattr(parent,n1,fused); setattr(parent,n2,nn.Identity())
    return model
