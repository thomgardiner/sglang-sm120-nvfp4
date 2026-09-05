import torch
from torch.profiler import profile, ProfilerActivity
from flashinfer import mm_fp4, fp4_quantize
dev = "cuda"; BW = 1.792e12
def timeit(fns, iters=30):
    for f in fns: f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns: f()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / (iters * len(fns)) * 1000
def knames(fn):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(); torch.cuda.synchronize()
    return sorted({e.name[:70] for e in prof.events() if e.device_type.name == "CUDA" and "Memcpy" not in e.name})
gs = torch.tensor(1.0, device=dev); alpha = torch.tensor(1.0, device=dev)
for M in (1, 9):
  for tag, N, K in [("gate_up", 34816, 5120), ("down", 5120, 17408)]:
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) for _ in range(8)]
    wq = [fp4_quantize(w, gs) for w in ws]
    xq, xs = fp4_quantize(x, gs)
    ref = None
    for be in ("cutlass", "cudnn", "cutedsl", "auto"):
        try:
            fns = [lambda a=a, b=b, be=be: mm_fp4(xq, a.t(), xs, b.t(), alpha, torch.bfloat16, backend=be) for a, b in wq]
            us = timeit(fns)
            out = fns[0]()
            if ref is None: ref = out
            diff = (out.float() - ref.float()).abs().max().item()
            print(f"M={M} {tag:8s} {be:8s} {us:7.1f} us {100*N*K*0.5625/us/1e6/(BW/1e12):5.1f}% HBM  maxabs_vs_cutlass={diff:.4g}  kernels={knames(fns[0])}", flush=True)
        except Exception as ex:
            print(f"M={M} {tag} {be} fail {repr(ex)[:150]}", flush=True)
    del ws, wq; torch.cuda.empty_cache()
