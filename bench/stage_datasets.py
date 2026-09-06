import json, gzip, io, sys
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
out = Path("datasets"); out.mkdir(exist_ok=True)

def save(name, rows):
    p = out / f"{name}.jsonl"
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(name, len(rows), p)

# HumanEval: 164 prompts
p = hf_hub_download("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet", repo_type="dataset")
import pyarrow.parquet as pq
t = pq.read_table(p).to_pylist()
save("humaneval", [{"id": r["task_id"], "prompt": "Complete the following Python function. Return the full function.\n\n" + r["prompt"]} for r in t])

# GSM8K test: first 100
files = list_repo_files("openai/gsm8k", repo_type="dataset")
f = [x for x in files if x.startswith("main/test")][0]
p = hf_hub_download("openai/gsm8k", f, repo_type="dataset")
t = pq.read_table(p).to_pylist()[:100]
save("gsm8k", [{"id": i, "prompt": r["question"], "answer": r["answer"]} for i, r in enumerate(t)])

# MATH-500: first 100
files = list_repo_files("HuggingFaceH4/MATH-500", repo_type="dataset")
f = [x for x in files if x.endswith(".jsonl") or x.endswith(".parquet")][0]
p = hf_hub_download("HuggingFaceH4/MATH-500", f, repo_type="dataset")
rows = [json.loads(l) for l in open(p)] if f.endswith(".jsonl") else pq.read_table(p).to_pylist()
save("math500", [{"id": i, "prompt": r["problem"], "answer": r.get("answer")} for i, r in enumerate(rows[:100])])

# MT-Bench: 80 first-turn prompts
files = list_repo_files("HuggingFaceH4/mt_bench_prompts", repo_type="dataset")
f = [x for x in files if x.endswith(".parquet") or x.endswith(".jsonl")][0]
p = hf_hub_download("HuggingFaceH4/mt_bench_prompts", f, repo_type="dataset")
rows = [json.loads(l) for l in open(p)] if f.endswith(".jsonl") else pq.read_table(p).to_pylist()
save("mtbench", [{"id": r.get("prompt_id", i), "category": r.get("category"), "prompt": r["prompt"][0] if isinstance(r["prompt"], list) else r["prompt"]} for i, r in enumerate(rows)])
