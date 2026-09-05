import torch, sys
dev = "cuda"; BW = 1.792e12
COPIES = 8
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
def report(tag, us, wbytes):
    print(f"{tag:42s} {us:7.1f} us {wbytes/1e6:7.1f} MB {wbytes/us/1e6:5.2f} TB/s {100*wbytes/us/1e6/(BW/1e12):5.1f}% of HBM", flush=True)
try:
    from flashinfer import mm_fp4, fp4_quantize
    HAVE_FP4 = True
except Exception as ex:
    print("fp4 import fail", repr(ex)[:200]); HAVE_FP4 = False
shapes = [("gate_up N=34816 K=5120", 34816, 5120), ("down N=5120 K=17408", 5120, 17408),
          ("gdn qkvz N=16384 K=5120", 16384, 5120), ("gdn out N=5120 K=6144", 5120, 6144),
          ("lm_head N=248320 K=5120", 248320, 5120)]
only = sys.argv[1] if len(sys.argv) > 1 else "all"
for M in (1, 9, 16):
    print(f"\n=== M={M} (cold: {COPIES} weight copies rotated)")
    for tag, N, K in shapes:
        big = "lm_head" in tag
        copies = 2 if big else COPIES
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) for _ in range(copies)]
        if only in ("all", "bf16") and not big:
            report(f"bf16 mm {tag}", timeit([lambda w=w: x @ w.t() for w in ws]), N * K * 2)
        if only in ("all", "fp8"):
            xq = x.to(torch.float8_e4m3fn); s1 = torch.tensor(1.0, device=dev)
            wqs = [w.to(torch.float8_e4m3fn).t() for w in ws]
            report(f"fp8 _scaled_mm {tag}", timeit([lambda wq=wq: torch._scaled_mm(xq, wq, scale_a=s1, scale_b=s1, out_dtype=torch.bfloat16) for wq in wqs]), N * K)
            if M == 1:
                try:
                    from sglang.kernels.ops.gemm.sm120_fp8_gemv import sm120_fp8_gemv, use_sm120_fp8_gemv
                    if use_sm120_fp8_gemv(1, N, K):
                        wc = [w.to(torch.float8_e4m3fn).contiguous() for w in ws]; al = torch.tensor([1.0], device=dev)
                        report(f"fp8 sm120_gemv {tag}", timeit([lambda w=w: sm120_fp8_gemv(xq, w, al) for w in wc]), N * K)
                    else: print("sm120_gemv not eligible", tag)
                except Exception as ex: print("gemv fail", repr(ex)[:120])
            del wqs
        if only in ("all", "fp4") and HAVE_FP4:
            try:
                gs = torch.tensor(1.0, device=dev); alpha = torch.tensor(1.0, device=dev)
                wq4s = [fp4_quantize(w, gs) for w in ws]
                xq4, xs4 = fp4_quantize(x, gs)
                fns = [lambda wq4=wq4, ws4=ws4: mm_fp4(xq4, wq4.t(), xs4, ws4.t(), alpha, torch.bfloat16, backend="cutlass") for wq4, ws4 in wq4s]
                report(f"nvfp4 mm_fp4 cutlass {tag}", timeit(fns), N * K * 0.5625)
                for be in ("cudnn", "trtllm", "auto"):
                    try:
                        fns = [lambda wq4=wq4, ws4=ws4, be=be: mm_fp4(xq4, wq4.t(), xs4, ws4.t(), alpha, torch.bfloat16, backend=be) for wq4, ws4 in wq4s]
                        report(f"nvfp4 mm_fp4 {be} {tag}", timeit(fns), N * K * 0.5625)
                    except Exception as ex: print(f"fp4 {be} fail", repr(ex)[:100])
                del wq4s
            except Exception as ex:
                print("fp4 fail", tag, repr(ex)[:200])
        del ws; torch.cuda.empty_cache()
