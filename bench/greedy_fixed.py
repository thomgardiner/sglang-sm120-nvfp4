import json, sys, time, urllib.request, hashlib, statistics
PORT, OUT = sys.argv[1], sys.argv[2]
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
import os
KEY = os.environ.get("SGLANG_API_KEY", "")
PROMPTS = {
    "mill": "Write a detailed walkthrough of a mill cellar search: ten short paragraphs, no list, keep going.",
    "math": "Let f(x) = x^3 - 6x^2 + 11x - 6. Find all real roots, then compute the integral of f from 1 to 3, and finally determine the x where f has its local maximum. Show every step.",
    "code": "Write a Rust function that parses an RFC 3339 timestamp into a Unix epoch in seconds without using any external crate. Handle timezone offsets and fractional seconds. Then write three test cases.",
}
def run(name, prompt):
    body = {"model": "qwen3.8-27b", "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": 1200, "seed": 1,
            "chat_template_kwargs": {"enable_thinking": True}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter(); tf = None; usage = {}; text = []
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            o = json.loads(d)
            if o.get("usage"): usage = o["usage"]
            delta = (o.get("choices") or [{}])[0].get("delta") or {}
            piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if piece:
                text.append(piece)
                if tf is None: tf = time.perf_counter()
    t1 = time.perf_counter()
    comp = usage.get("completion_tokens", 0)
    full = "".join(text)
    return {"prompt": name, "completion": comp, "sha": hashlib.sha256(full.encode()).hexdigest()[:16],
            "ttft_s": round(tf - t0, 3) if tf else None, "decode_s": round(t1 - tf, 3) if tf else None,
            "decode_tok_s": round(comp / (t1 - tf), 1) if tf else 0}
rows = []
for rep in range(2):
    for name, p in PROMPTS.items():
        r = run(name, p); r["rep"] = rep; rows.append(r); print(json.dumps(r), flush=True)
json.dump(rows, open(OUT, "w"), indent=1)
