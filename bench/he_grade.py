import glob, json, re, subprocess, sys, tempfile, os
tag = sys.argv[1]
f = sorted(glob.glob(f"lmeval/nothink-{tag}/qwen3.8-27b/samples_humaneval_instruct_*.jsonl"))[-1]
rows = [json.loads(l) for l in open(f)]
FENCE = re.compile(r"```(?:python)?\s*\n(.*?)(?:\n```|$)", re.S)

def code_of(resp):
    m = FENCE.findall(resp)
    return max(m, key=len) if m else resp

passed = 0; failed = []
for r in rows:
    resp = r["resps"][0][0]
    prog = code_of(resp) + "\n\n" + r["doc"]["test"] + f"\n\ncheck({r['doc']['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as t:
        t.write(prog); path = t.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, timeout=15)
        ok = p.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
    os.unlink(path)
    passed += ok
    if not ok: failed.append(r["doc"]["task_id"])
print(f"HE {tag}: pass@1 = {passed}/{len(rows)} = {passed/len(rows):.4f}")
print("failed:", failed[:12], "..." if len(failed) > 12 else "")
json.dump({"tag": tag, "n": len(rows), "passed": passed, "pass_at_1": passed / len(rows), "failed": failed, "samples_file": f},
          open(f"lmeval/humaneval_{tag}.json", "w"), indent=1)
