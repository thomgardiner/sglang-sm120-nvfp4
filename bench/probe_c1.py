import json, os, sys, time, urllib.request

port, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]
key = os.environ.get("SGLANG_API_KEY", "qwen-local-tg")

PROMPTS = {
    "prose": "Explain in plain prose, ten paragraphs, how a water mill turns grain into flour. No lists.",
    "code": "Write a Python module that parses a CSV of transactions, groups them by month, and prints a table of totals with a running balance. Full code, with a small test at the bottom.",
    "math": "A tank fills at 3 L/min through one pipe and drains at 1.2 L/min through another. It starts with 40 L and holds 400 L. Work out, step by step, when it is full, then how long a second drain at 0.8 L/min would delay that.",
}


def run(name, prompt):
    body = {
        "model": "qwen3.8-27b",
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 512,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        "http://127.0.0.1:%s/v1/chat/completions" % port,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
    )
    steps = 0
    usage = None
    first = None
    t0 = time.time()
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
    return {
        "prompt": name,
        "prompt_tokens": usage["prompt_tokens"],
        "tokens": toks,
        "steps": steps,
        "ttft_s": round(first - t0, 3),
        "decode_s": round(dec, 3),
        "step_ms": round(1000 * dec / steps, 2),
        "tok_per_step": round(toks / steps, 3),
        "tok_s": round(toks / dec, 1),
    }


run("warm", PROMPTS["prose"])
rows = []
for name, prompt in PROMPTS.items():
    for rep in range(2):
        r = run(name, prompt)
        r["rep"] = rep
        rows.append(r)
        print(
            "%s %s rep%s tok/s=%s step=%s tok/step=%s"
            % (tag, name, rep, r["tok_s"], r["step_ms"], r["tok_per_step"]),
            flush=True,
        )
json.dump({"tag": tag, "rows": rows}, open(out, "w"), indent=1)
print("WROTE", out, flush=True)
