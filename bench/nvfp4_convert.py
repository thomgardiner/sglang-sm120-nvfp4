import json, re, shutil, struct, sys, torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

SRC_BF16 = Path(sys.argv[1]); SRC_Q = Path(sys.argv[2]); DST = Path(sys.argv[3])
MODE = sys.argv[4] if len(sys.argv) > 4 else "convert"
E4M3_MAX = 448.0; E2M1_MAX = 6.0; BLOCK = 16
E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def e2m1_encode(x):
    sign = (x < 0).to(torch.uint8)
    a = x.abs()
    idx = torch.bucketize(a, (E2M1_GRID[1:] + E2M1_GRID[:-1]) / 2)
    return (sign << 3) | idx.to(torch.uint8)

def quantize_nvfp4(w_bf16, global_amax=None):
    w = w_bf16.float()
    N, K = w.shape
    assert K % BLOCK == 0
    amax = w.abs().amax() if global_amax is None else global_amax
    scale2 = (amax / (E4M3_MAX * E2M1_MAX)).float()
    wb = w.view(N, K // BLOCK, BLOCK)
    bamax = wb.abs().amax(dim=-1)
    bscale = (bamax / E2M1_MAX) / scale2
    bscale_e4m3 = bscale.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    bscale_f = bscale_e4m3.float().clamp(min=1e-12)
    q = (wb / (bscale_f * scale2).unsqueeze(-1)).clamp(-E2M1_MAX, E2M1_MAX)
    codes = e2m1_encode(q).view(N, K)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, bscale_e4m3, scale2

def dequant_nvfp4(packed, bscale_e4m3, scale2):
    lo = packed & 0xF; hi = packed >> 4
    codes = torch.stack([lo, hi], dim=-1).view(packed.shape[0], -1)
    mag = E2M1_GRID[(codes & 7).long()]
    sign = torch.where((codes >> 3).bool(), -1.0, 1.0)
    vals = (mag * sign).view(packed.shape[0], -1, BLOCK)
    return (vals * (bscale_e4m3.float() * scale2).unsqueeze(-1)).view(packed.shape[0], -1)

def load_index(d):
    return json.load(open(d / "model.safetensors.index.json"))["weight_map"]

def get(d, idx, name, cache={}):
    f = idx[name]
    key = (str(d), f)
    if key not in cache:
        cache[key] = safe_open(str(d / f), framework="pt")
    return cache[key].get_tensor(name)

qidx = load_index(SRC_Q); bidx = load_index(SRC_BF16)
bkey = {k.replace("model.language_model.", "model."): k for k in bidx}

def bf16_name(qname):
    for cand in (qname, qname.replace("model.language_model.", "model."), qname.replace("model.language_model.", "")):
        if cand in bidx: return cand
    raise KeyError(qname)

if MODE == "check":
    for layer in ("model.language_model.layers.0.mlp.gate_proj", "model.language_model.layers.5.mlp.down_proj"):
        w = get(SRC_BF16, bidx, bf16_name(layer + ".weight"))
        p, s, s2 = quantize_nvfp4(w)
        p0 = get(SRC_Q, qidx, layer + ".weight"); s0 = get(SRC_Q, qidx, layer + ".weight_scale"); s20 = get(SRC_Q, qidx, layer + ".weight_scale_2")
        deq_mine = dequant_nvfp4(p, s, s2); deq_ref = dequant_nvfp4(p0, s0, s20)
        err_mine = (deq_mine - w.float()).pow(2).mean().sqrt().item(); err_ref = (deq_ref - w.float()).pow(2).mean().sqrt().item()
        print(f"{layer.split('.',3)[-1]}: scale2 mine={s2.item():.6g} ref={s20.item():.6g} | byte match {(p==p0).float().mean().item():.4f} | blockscale match {(s.view(torch.uint8)==s0.view(torch.uint8)).float().mean().item():.4f} | rmse mine {err_mine:.5g} ref {err_ref:.5g} | w rms {w.float().pow(2).mean().sqrt().item():.5g}")
    sys.exit(0)

hq = json.load(open(SRC_Q / "hf_quant_config.json"))
layers = hq["quantization"]["quantized_layers"]
fp8_layers = sorted(k for k, v in layers.items() if v["quant_algo"] == "FP8")
groups = {}
for k in fp8_layers:
    base, leaf = k.rsplit(".", 1)
    g = {"in_proj_qkv": "qkvz", "in_proj_z": "qkvz", "q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv"}.get(leaf, leaf)
    groups.setdefault((base, g), []).append(k)
print(f"{len(fp8_layers)} FP8 layers in {len(groups)} fusion groups")

DST.mkdir(parents=True, exist_ok=True)
new_tensors = {}; drop = set(); stats = []
for (base, g), names in groups.items():
    ws = {n: get(SRC_BF16, bidx, bf16_name(n + ".weight")) for n in names}
    gamax = max(w.abs().amax() for w in ws.values())
    in_scales = [get(SRC_Q, qidx, n + ".input_scale").float() for n in names]
    in_scale_nvfp4 = (max(in_scales) / E2M1_MAX).reshape(())
    for n in names:
        p, s, s2 = quantize_nvfp4(ws[n], global_amax=gamax)
        new_tensors[n + ".weight"] = p; new_tensors[n + ".weight_scale"] = s
        new_tensors[n + ".weight_scale_2"] = s2.reshape(()); new_tensors[n + ".input_scale"] = in_scale_nvfp4.clone()
        drop.update({n + ".weight", n + ".weight_scale", n + ".input_scale"})
        deq = dequant_nvfp4(p, s, s2); w = ws[n].float()
        stats.append((n, (deq - w).pow(2).mean().sqrt().item() / w.pow(2).mean().sqrt().item()))
        layers[n] = {"quant_algo": "NVFP4", "group_size": 16}
print("relative rmse: mean %.4f max %.4f (%s)" % (sum(s for _, s in stats) / len(stats), max(s for _, s in stats), max(stats, key=lambda x: x[1])[0]))

shards = {}
for name, f in qidx.items():
    shards.setdefault(f, []).append(name)
new_index = {"metadata": {}, "weight_map": {}}
for f, names in sorted(shards.items()):
    out = {}
    fh = safe_open(str(SRC_Q / f), framework="pt")
    for n in names:
        if n in drop: continue
        out[n] = fh.get_tensor(n)
    for n in [x for x in new_tensors if qidx.get(x.replace(".weight_scale_2", ".weight").replace(".weight_scale", ".weight").replace(".input_scale", ".weight")) == f]:
        out[n] = new_tensors[n]
    save_file(out, str(DST / f), metadata={"format": "pt"})
    for n in out: new_index["weight_map"][n] = f
    print("wrote", f, len(out))
new_index["metadata"]["total_size"] = sum((DST / f).stat().st_size for f in shards)
json.dump(new_index, open(DST / "model.safetensors.index.json", "w"), indent=2)
json.dump(hq, open(DST / "hf_quant_config.json", "w"), indent=4)
for f in SRC_Q.iterdir():
    if f.is_file() and f.name not in shards and f.name not in ("model.safetensors.index.json", "hf_quant_config.json"):
        shutil.copy(f, DST / f.name)
cfg = json.load(open(DST / "config.json"))
qc = cfg.get("quantization_config")
if qc and "config_groups" in qc:
    fp8_group = [g for g, v in qc["config_groups"].items() if v.get("weights", {}).get("num_bits") == 8]
    for g in fp8_group:
        qc["config_groups"][g]["targets"] = []
    fp4_group = [g for g, v in qc["config_groups"].items() if v.get("weights", {}).get("num_bits") == 4]
    if fp4_group:
        qc["config_groups"][fp4_group[0]]["targets"] = sorted(set(qc["config_groups"][fp4_group[0]].get("targets", [])) | set(fp8_layers))
    if "quantized_layers" in qc:
        for n in fp8_layers: qc["quantized_layers"][n] = {"quant_algo": "NVFP4", "group_size": 16}
    json.dump(cfg, open(DST / "config.json", "w"), indent=2)
    print("config.json quantization_config updated; fp8 groups emptied:", fp8_group, "| quantized_layers NVFP4:", sum(1 for v in qc.get("quantized_layers", {}).values() if v["quant_algo"] == "NVFP4"))
print("done", DST)
