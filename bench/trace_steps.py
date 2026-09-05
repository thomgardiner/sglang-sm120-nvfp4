import gzip, json, sys, collections
ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
steps = sorted([e for e in ev if e.get("name","").startswith("step[")], key=lambda e: e["ts"])
kern = sorted([e for e in ev if e.get("cat") == "kernel"], key=lambda e: e["ts"])
print("steps:", collections.Counter(e["name"].split(" ")[0] for e in steps))
for e in steps[:6]: print(f'{e["name"]:36s} dur_ms={e["dur"]/1000:.2f}')
# decode window = after the EXTEND step ends
ext = [e for e in steps if "EXTEND" in e["name"]][0]
dec0 = ext["ts"] + ext["dur"]
dk = [k for k in kern if k["ts"] > dec0]
wall = (dk[-1]["ts"] + dk[-1]["dur"] - dk[0]["ts"]) / 1000
busy = sum(k["dur"] for k in dk) / 1000
n_ver = sum(1 for e in steps if "VERIFY" in e["name"] and e["ts"] > dec0)
print(f"decode window: wall={wall:.1f}ms busy={busy:.1f}ms idle={wall-busy:.1f}ms verify_steps={n_ver} per_step_wall={wall/max(n_ver,1):.2f}ms")
names = collections.defaultdict(float); cnt = collections.Counter()
for k in dk:
    nm = k["name"][:60]; names[nm] += k["dur"]/1000; cnt[nm] += 1
print("\nper decode step (ms, calls/step, kernel)")
for nm, ms in sorted(names.items(), key=lambda x: -x[1])[:16]:
    print(f"{ms/n_ver:6.2f} {cnt[nm]/n_ver:7.1f}  {nm}")
# gaps: list largest idle gaps between consecutive kernels in decode window, and which CPU annotation covers them
gaps = []
for a, b in zip(dk, dk[1:]):
    g = b["ts"] - (a["ts"] + a["dur"])
    if g > 300: gaps.append((g, a["ts"] + a["dur"], a["name"][:40], b["name"][:40]))
gaps.sort(reverse=True)
print(f"\ngaps >0.3ms: n={len(gaps)} total={sum(g[0] for g in gaps)/1000:.1f}ms ; per step {sum(g[0] for g in gaps)/1000/n_ver:.2f}ms")
ann = [e for e in ev if e.get("cat") in ("user_annotation","python_function","cpu_op") and e.get("dur",0) > 200]
def cover(ts):
    c = [e for e in ann if e["ts"] <= ts <= e["ts"] + e["dur"] and e["dur"] < 20000]
    c.sort(key=lambda e: e["dur"])
    return [e["name"].split("/")[-1][:60] for e in c[:3]]
for g in gaps[:10]:
    print(f"  {g[0]/1000:5.2f}ms after {g[2]} -> {g[3]} :: {cover(g[1] + g[0]/2)}")
