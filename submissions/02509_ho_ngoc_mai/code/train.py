"""One reproducible training entry point for every DeepWeeds experiment."""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import argparse, copy, json, math, random, time
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.metrics import f1_score
from dataset import load_split, check_split, build_transforms, make_loader
from model import build_model, param_groups, count_params, count_gmacs
from losses import build_criterion, class_weights, mix_batch, mixed_loss

@dataclass
class Config:
    exp_id:str="T00"; seed:int=0; fold:int=0; backbone:str="resnet50"; init:str="finetune"; drop_rate:float=0.
    img_size:int=224; aug:str="basic"; sampler:str|None=None; mix:str|None=None; mix_alpha:float=1.
    loss:str="ce"; label_smoothing:float=0.; focal_gamma:float=2.; class_weight_beta:float|None=None
    epochs:int=12; batch_size:int=64; lr_backbone:float=1e-4; lr_head:float=1e-3; weight_decay:float=.05; warmup_epochs:float=1.; ema_decay:float|None=None; amp:bool=True; num_workers:int=2
    images_dir:str="data"; labels_dir:str="data/labels"; out_dir:str="runs"; pred_dir:str="predictions"; save_test_predictions:bool=False
def run_dir(c): return Path(c.out_dir)/c.exp_id/f"seed{c.seed}"
def pred_path(c,split): return Path(c.pred_dir)/f"{c.exp_id}_seed{c.seed}_{split}.csv"
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def build_optimizer(m,c): return torch.optim.AdamW(param_groups(m,c.lr_backbone,c.lr_head,c.weight_decay))
def build_scheduler(opt,c,steps):
    total=max(1,c.epochs*steps); warm=max(1,int(c.warmup_epochs*steps))
    return torch.optim.lr_scheduler.LambdaLR(opt,lambda s:(s+1)/warm if s<warm else .5*(1+math.cos(math.pi*(s-warm)/max(1,total-warm))))
class EMA:
    def __init__(self,m,decay): self.decay=decay; self.model=copy.deepcopy(m).eval(); [setattr(p,'requires_grad',False) for p in self.model.parameters()]
    @torch.no_grad()
    def update(self,m):
        for e,p in zip(self.model.state_dict().values(),m.state_dict().values()):
            if e.is_floating_point(): e.mul_(self.decay).add_(p.detach(),alpha=1-self.decay)
            else: e.copy_(p)
def train_one_epoch(m,loader,criterion,opt,sched,scaler,c,device,ema=None):
    m.train(); loss_sum=0.
    if c.init=="frozen":
        for mod in m.modules():
            if isinstance(mod,nn.modules.batchnorm._BatchNorm): mod.eval()
    for x,y,_ in loader:
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True); targets=y
        if c.mix: x,targets=mix_batch(x,y,c.mix_alpha,c.mix)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=c.amp and device.type=="cuda"): out=m(x); loss=mixed_loss(criterion,out,targets) if c.mix else criterion(out,y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); loss_sum+=loss.item()*len(y)
        if ema: ema.update(m)
    return {"train_loss":loss_sum/len(loader.dataset),"lr":opt.param_groups[0]["lr"]}
@torch.inference_mode()
def evaluate(m,loader,criterion,device):
    m.eval(); names=[]; yy=[]; logs=[]; total=0.
    for x,y,n in loader:
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True); z=m(x); total+=criterion(z,y).item()*len(y); names.extend(n); yy.extend(y.cpu().tolist()); logs.append(z.float().cpu())
    return names,np.asarray(yy),torch.cat(logs).numpy(),total/len(loader.dataset)
def plot_curves(hist,path,title):
    import matplotlib.pyplot as plt
    d=pd.DataFrame(hist); fig,ax=plt.subplots(1,2,figsize=(10,3)); ax[0].plot(d.epoch,d.train_loss,label="train"); ax[0].plot(d.epoch,d.val_loss,label="val"); ax[0].legend(); ax[0].set_title("loss"); ax[1].plot(d.epoch,d.macro_f1); ax[1].set_title("val macro-F1"); fig.suptitle(title); Path(path).parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=160,bbox_inches="tight"); plt.close(fig)
def run(c:Config):
    from eval import save_predictions
    set_seed(c.seed); rd=run_dir(c); rd.mkdir(parents=True,exist_ok=True); Path(c.pred_dir).mkdir(parents=True,exist_ok=True); (rd/'config.json').write_text(json.dumps(asdict(c),indent=2))
    tr,va,te=load_split(c.labels_dir,c.fold); check_split(tr,va,te,c.images_dir)
    tl=make_loader(tr,c.images_dir,build_transforms(True,c.img_size,c.aug),c.batch_size,True,c.sampler,c.num_workers); vl=make_loader(va,c.images_dir,build_transforms(False,c.img_size),c.batch_size,False,None,c.num_workers)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); m=build_model(c.backbone,init=c.init,drop_rate=c.drop_rate).to(device); weight=class_weights(tr.Label.value_counts().sort_index(),c.class_weight_beta or 0).to(device) if c.loss=="ce_weighted" else None; criterion=build_criterion(c.loss,smoothing=c.label_smoothing,gamma=c.focal_gamma,weight=weight).to(device); opt=build_optimizer(m,c); sch=build_scheduler(opt,c,len(tl)); scaler=torch.amp.GradScaler(device.type,enabled=c.amp and device.type=="cuda"); ema=EMA(m,c.ema_decay) if c.ema_decay else None; hist=[]; best=(-1,None)
    for epoch in range(c.epochs):
        t=time.perf_counter(); row={"epoch":epoch+1,**train_one_epoch(m,tl,criterion,opt,sch,scaler,c,device,ema)}; eval_m=ema.model if ema else m; names,y,logits,vloss=evaluate(eval_m,vl,criterion,device); row.update(val_loss=vloss,macro_f1=f1_score(y,logits.argmax(1),average="macro"),top1=float((y==logits.argmax(1)).mean()),seconds=time.perf_counter()-t); hist.append(row)
        if row["macro_f1"]>best[0]: best=(row["macro_f1"],copy.deepcopy(eval_m.state_dict())); torch.save({"state_dict":best[1],"epoch":epoch+1},rd/'best.pt')
    m.load_state_dict(best[1]); names,y,logits,_=evaluate(m,vl,criterion,device); probs=torch.softmax(torch.tensor(logits),1).numpy(); np.save(rd/'val_logits.npy',logits); save_predictions(pred_path(c,'val'),names,y,probs); pd.DataFrame(hist).to_csv(rd/'history.csv',index=False); curve_dir=Path(c.out_dir).parent/'curves'; plot_curves(hist,curve_dir/f"{c.exp_id}_{c.backbone}.png",f"{c.exp_id}: {c.backbone}")
    if c.save_test_predictions:
        testl=make_loader(te,c.images_dir,build_transforms(False,c.img_size),c.batch_size,False,None,c.num_workers); names,y,logits,_=evaluate(m,testl,criterion,device); np.save(rd/'test_logits.npy',logits); save_predictions(pred_path(c,'test'),names,y,torch.softmax(torch.tensor(logits),1).numpy())
    return {"exp_id":c.exp_id,"seed":c.seed,"best_epoch":int(np.argmax([r['macro_f1'] for r in hist])+1),"macro_f1_val":best[0],"top1_val":max(r['top1'] for r in hist),"params_m":count_params(m),"gmac":count_gmacs(m,c.img_size),"train_seconds_per_epoch":float(np.mean([r['seconds'] for r in hist]))}
def parse_overrides(pairs):
    types={f.name:f.type for f in fields(Config)}; out={}
    for p in pairs:
        k,v=p.split('=',1)
        if k not in types: raise KeyError(f"Unknown Config field: {k}")
        out[k]=None if v.lower()=="none" else ({'true':True,'false':False}.get(v.lower(), v))
        if types[k] is int: out[k]=int(out[k])
        elif types[k] is float: out[k]=float(out[k])
    return out
def main():
    p=argparse.ArgumentParser(); p.add_argument('--set',nargs='*',default=[]); a=p.parse_args(); print(run(Config(**parse_overrides(a.set))))
if __name__=='__main__': main()
