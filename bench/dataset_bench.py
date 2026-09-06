import json, sys, time, urllib.request, subprocess, re, statistics, hashlib
PORT, CONT, DATASET, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
THINK = (sys.argv[5] if len(sys.argv) > 5 else "on") == "on"
MAXTOK = int(sys.argv[6]) if len(sys.argv) > 6 else 1024
LIMIT = int(sys.argv[7]) if len(sys.argv) > 7 else 10**9
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
import os
KEY = os.environ.get("SGLANG_API_KEY", "")
rows_in = [json.loads(l) for l in open(f"datasets/{DATASET}.jsonl")][:LIMIT]

def run(r):
    since = time.strftime("%Y-%m-%dT%H:%M:%S")
    body = {"model": "qwen3.8-27b", "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": r["prompt"]}], "temperature": 0.0, "max_tokens": MAXTOK,
            "chat_template_kwargs": {"enable_thinking": THINK}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t0 = time.perf_counter(); tf = None; usage = {}; text = []
    with urllib.request.urlopen(req, timeout=600) as resp:
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
    t1 = time.perf_counter(); comp = usage.get("completion_tokens", 0)
    time.sleep(1.2)
    log = subprocess.run(["docker", "logs", "--since", since, CONT], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    pairs = [(float(a), float(t)) for a, t in re.findall(r"accept len: ([\d.]+), accept rate: [\d.]+.*?gen throughput \(token/s\): ([\d.]+)", log) if float(t) > 50]
    full = "".join(text)
    return {"id": r["id"], "completion": comp, "decode_s": round(t1 - tf, 3) if tf else None,
            "decode_tok_s": round(comp / (t1 - tf), 1) if tf and comp else None,
            "accept": round(statistics.mean(a for a, _ in pairs), 3) if pairs else None,
            "accept_batches": len(pairs), "sha": hashlib.sha256(full.encode()).hexdigest()[:16],
            "finish_tokens": comp, "answer_tail": full[-300:]}

out = open(OUT, "w"); res = []
for i, r in enumerate(rows_in):
    try:
        x = run(r)
    except Exception as e:
        x = {"id": r["id"], "error": repr(e)[:200]}
    res.append(x); out.write(json.dumps(x) + "\n"); out.flush()
    if i % 10 == 0: print(i, json.dumps({k: x.get(k) for k in ("id", "completion", "decode_tok_s", "accept")}), flush=True)
ok = [x for x in res if x.get("decode_tok_s") and x.get("accept")]
tot_tok = sum(x["completion"] for x in ok); tot_s = sum(x["decode_s"] for x in ok)
print(f"SUMMARY {DATASET} think={'on' if THINK else 'off'} n={len(ok)} tokens={tot_tok} "
      f"tok/s(total)={tot_tok/tot_s:.1f} mean_prompt_tok/s={statistics.mean(x['decode_tok_s'] for x in ok):.1f} "
      f"accept(mean)={statistics.mean(x['accept'] for x in ok):.3f} "
      f"accept(token-weighted)={sum(x['accept']*x['completion'] for x in ok)/tot_tok:.3f}", flush=True)
