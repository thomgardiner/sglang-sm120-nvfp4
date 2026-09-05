import json, re, shutil, sys, torch
from pathlib import Path
from safetensors.torch import load_file, save_file

src = Path(sys.argv[1]); dst = Path(sys.argv[2]); STRATEGY = sys.argv[3] if len(sys.argv) > 3 else "channel"
dst.mkdir(parents=True, exist_ok=True)
QUANT = re.compile(r"^layers\.\d+\.(mlp\.(gate|up|down)_proj|self_attn\.o_proj)\.weight$")
FMAX = torch.finfo(torch.float8_e4m3fn).max

tensors = load_file(str(src / "model.safetensors"))
out = {}; n_q = 0; bytes_before = 0; bytes_after = 0
for name, t in tensors.items():
    bytes_before += t.numel() * t.element_size()
    if QUANT.match(name):
        w = t.float()
        scale = (w.abs().amax(dim=1, keepdim=True) if STRATEGY == "channel" else w.abs().amax().reshape(1, 1)).clamp(min=1e-12) / FMAX
        q = (w / scale).clamp(-FMAX, FMAX).to(torch.float8_e4m3fn)
        out[name] = q
        out[name.replace(".weight", ".weight_scale")] = (scale.to(torch.float32) if STRATEGY == "channel" else scale.reshape(()).to(torch.float32))
        bytes_after += q.numel() + scale.numel() * 4
        n_q += 1
    else:
        out[name] = t
        bytes_after += t.numel() * t.element_size()
save_file(out, str(dst / "model.safetensors"), metadata={"format": "pt"})

cfg = json.load(open(src / "config.json"))
cfg["quantization_config"] = {
    "quant_method": "compressed-tensors",
    "format": "float-quantized",
    "quantization_status": "compressed",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 8, "type": "float", "strategy": STRATEGY, "symmetric": True, "dynamic": False},
            "input_activations": {"num_bits": 8, "type": "float", "strategy": "token" if STRATEGY == "channel" else "tensor", "symmetric": True, "dynamic": True},
        }
    },
    "ignore": [
        "re:.*self_attn\\.q_proj.*", "re:.*self_attn\\.k_proj.*", "re:.*self_attn\\.v_proj.*",
        "re:.*qkv_proj.*", "re:.*fc.*", "re:.*candidate_selector.*", "re:.*conv.*", "re:.*norm.*", "lm_head",
    ],
}
json.dump(cfg, open(dst / "config.json", "w"), indent=2)
for f in src.iterdir():
    if f.name not in ("model.safetensors", "config.json") and f.is_file():
        shutil.copy(f, dst / f.name)
print(f"quantized {n_q} tensors; {bytes_before/1e9:.2f} GB -> {bytes_after/1e9:.2f} GB")
