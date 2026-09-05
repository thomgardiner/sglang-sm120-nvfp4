import gzip, json, sys, re, collections
path = sys.argv[1]
ev = json.load(gzip.open(path))["traceEvents"]
kern = [e for e in ev if e.get("cat") == "kernel"]
kern.sort(key=lambda e: e["ts"])
t0, t1 = kern[0]["ts"], kern[-1]["ts"] + kern[-1]["dur"]
wall = (t1 - t0) / 1000
busy = sum(e["dur"] for e in kern) / 1000
print(f"kernels={len(kern)} wall_ms={wall:.1f} gpu_busy_ms={busy:.1f} idle_ms={wall-busy:.1f}")
def bucket(n):
    n2 = n.lower()
    if "cutlass" in n2 or "gemm" in n2 or "nvfp4" in n2 or "fp4" in n2 or "block_scaled" in n2 or "sm120" in n2: return "gemm"
    if "gdn" in n2 or "recurrent" in n2 or "delta" in n2 or "kda" in n2 or "chunk_" in n2 or "gated" in n2: return "gdn"
    if "conv1d" in n2 or "causal_conv" in n2: return "conv1d"
    if "attention" in n2 or "flashinfer" in n2 or "decode_kernel" in n2 or "prefill" in n2 or "fmha" in n2 or "batchdecode" in n2 or "batchprefill" in n2: return "attention"
    if "norm" in n2: return "norm"
    if "silu" in n2 or "act_and_mul" in n2 or "activation" in n2: return "act"
    if "elementwise" in n2 or "vectorized" in n2 or "copy" in n2 or "memcpy" in n2 or "fill" in n2 or "index" in n2 or "gather" in n2 or "scatter" in n2 or "cat" in n2: return "elementwise/copy"
    if "sampl" in n2 or "topk" in n2 or "softmax" in n2 or "argmax" in n2 or "multinomial" in n2: return "sampling"
    if "rotary" in n2 or "rope" in n2: return "rope"
    if "embed" in n2: return "embed"
    return "other"
by = collections.defaultdict(float); cnt = collections.Counter()
for e in kern:
    b = bucket(e["name"]); by[b] += e["dur"] / 1000; cnt[b] += 1
print("\nbucket           ms     pct   count")
for b, ms in sorted(by.items(), key=lambda x: -x[1]):
    print(f"{b:16s} {ms:7.1f} {100*ms/busy:5.1f}%  {cnt[b]}")
names = collections.defaultdict(float); nc = collections.Counter()
for e in kern: names[e["name"]] += e["dur"] / 1000; nc[e["name"]] += 1
print("\ntop kernels (ms, count, name)")
for n, ms in sorted(names.items(), key=lambda x: -x[1])[:28]:
    print(f"{ms:7.1f} {nc[n]:6d}  {n[:110]}")
# step boundaries: cpu events named like forward / verify / draft
cpu = [e for e in ev if e.get("cat") in ("cpu_op","user_annotation","python_function") and e.get("dur")]
ann = collections.defaultdict(float); ac = collections.Counter()
for e in cpu:
    n = e["name"]
    if any(k in n.lower() for k in ("draft", "verify", "forward", "sample", "graph", "replay", "scheduler", "step")):
        ann[n] += e["dur"]/1000; ac[n] += 1
print("\nannotations (ms total, count, name)")
for n, ms in sorted(ann.items(), key=lambda x: -x[1])[:25]:
    print(f"{ms:8.1f} {ac[n]:5d}  {n[:100]}")
