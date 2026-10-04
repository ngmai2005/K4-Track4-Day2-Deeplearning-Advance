"""GPU-synchronised inference latency measurements."""
from __future__ import annotations
import time, numpy as np, torch
def bench(fn,warmup=10,iters=100,sync=None):
    for _ in range(warmup): fn()
    values=[]
    for _ in range(iters):
        if sync: sync()
        t=time.perf_counter(); fn()
        if sync: sync()
        values.append((time.perf_counter()-t)*1000)
    return {"p50":float(np.percentile(values,50)),"p95":float(np.percentile(values,95)),"p99":float(np.percentile(values,99)),"mean":float(np.mean(values)),"n":iters}
def latency_report(model,batch_size,img_size,dtype="fp32",device="cuda",warmup=10,iters=100):
    dev=torch.device(device); model=model.eval().to(dev); x=torch.randn(batch_size,3,img_size,img_size,device=dev); use_amp=dtype=="amp"
    if dtype=="fp16": model=model.half(); x=x.half()
    if dtype not in {"fp32","amp","fp16"}: raise ValueError("dtype must be fp32, amp or fp16")
    def fn():
        with torch.inference_mode(), torch.autocast(device_type=dev.type,enabled=use_amp): model(x)
    sync=torch.cuda.synchronize if dev.type=="cuda" else None; r=bench(fn,warmup,iters,sync)
    return {**r,"gpu":torch.cuda.get_device_name(dev) if dev.type=="cuda" else "CPU","dtype":dtype,"batch":batch_size,"img_size":img_size,"images_per_s":batch_size/(r["p50"]/1000),"torch":torch.__version__}
def tta_latency(model,k_views,**kw):
    r=latency_report(model,**kw); r["k_views"]=k_views; r["estimated_total_p50"]=r["p50"]*k_views; return r
