import json, sys, time, random, urllib.request

port, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]
key = __import__("os").environ.get("SGLANG_API_KEY", "")

WORDS = "mill wheel grain stone river flour sack cart village season harvest water gear shaft beam roof oak iron rope ledger".split()
rng = random.Random(7)
filler = " ".join(rng.choice(WORDS) for _ in range(9000))

PROMPTS = {
    "prose": "Explain in plain prose, ten paragraphs, how a water mill turns grain into flour. No lists.",
    "code": "Write a Python module that parses a CSV of transactions, groups them by month, and prints a table of totals with a running balance. Full code, with a small test at the bottom.",
    "math": "A tank fills at 3 L/min through one pipe and drains at 1.2 L/min through another. It starts with 40 L and holds 400 L. Work out, step by step, when it is full, then how long a second drain at 0.8 L/min would delay that.",
    "prose_8k": filler + "\n\nIgnore the word list above. Explain in plain prose, ten paragraphs, how a water mill turns grain into flour. No lists.",
}

def run(name, prompt):
    body = {"model": "qwen3.8-27b", "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": 512, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    steps = 0; usage = None; first = None; t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                if first is None:
                    first = time.time()
                steps += 1
    end = time.time()
    toks = usage["completion_tokens"]
    dec = end - first
    return {"prompt": name, "prompt_tokens": usage["prompt_tokens"], "tokens": toks, "steps": steps,
            "ttft_s": round(first - t0, 3), "decode_s": round(dec, 3),
            "step_ms": round(1000 * dec / steps, 2), "tok_per_step": round(toks / steps, 3), "tok_s": round(toks / dec, 1)}

run("warm", PROMPTS["prose"])
rows = []
for name, p in PROMPTS.items():
    for rep in range(2):
        r = run(name, p); r["rep"] = rep; rows.append(r)
        print(f"{tag:22s} {name:9s} rep{rep} prompt={r['prompt_tokens']:5d} tok={r['tokens']:4d} steps={r['steps']:4d} step={r['step_ms']:6.2f} ms  tok/step={r['tok_per_step']:.2f}  tok/s={r['tok_s']:.1f}", flush=True)
json.dump({"tag": tag, "rows": rows}, open(out, "w"), indent=1)
