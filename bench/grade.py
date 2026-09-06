import json, re, sys
DATASET, RUN = sys.argv[1], sys.argv[2]
gold = {r["id"]: r for r in (json.loads(l) for l in open(f"datasets/{DATASET}.jsonl"))}
rows = [json.loads(l) for l in open(RUN)]

def norm_num(s):
    s = s.replace(",", "").replace("$", "").strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return m[-1] if m else None

def gold_answer(r):
    a = r["answer"]
    if DATASET == "gsm8k":
        return norm_num(a.split("####")[-1])
    return str(a).strip()

def pred_answer(tail):
    boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", tail)
    if boxed: return boxed[-1].strip()
    m = re.findall(r"(?:answer is|Answer:|answer:)\s*\**\s*([^\n*]+)", tail)
    if m: return m[-1].strip().rstrip(".")
    return tail.strip().splitlines()[-1] if tail.strip() else ""

def match(p, g):
    if p is None or g is None: return False
    p2, g2 = p.replace(" ", "").replace("\\!", "").rstrip("."), g.replace(" ", "")
    if p2 == g2: return True
    pn, gn = norm_num(p2), norm_num(g2)
    if pn is not None and gn is not None and re.fullmatch(r"-?\d+(?:\.\d+)?", g2.replace("\\", "")):
        try: return abs(float(pn) - float(gn)) < 1e-6
        except ValueError: return False
    return False

n = ok = trunc = 0
for r in rows:
    g = gold.get(r["id"]);
    if g is None or "answer_tail" not in r: continue
    n += 1
    if r.get("completion", 0) >= 1024: trunc += 1
    ok += match(pred_answer(r["answer_tail"]), gold_answer(g))
print(f"{DATASET} {RUN.split('/')[-1]}: n={n} correct={ok} acc={ok/n:.3f} hit_max_tokens={trunc}")
