# Benchmarks

One RTX 5090, SGLang image `lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe`, target Qwen3.8-27B NVFP4 (ModelOpt), DFlash2 draft, 8 draft tokens, `--max-running-requests 1 --cuda-graph-max-bs-decode 1 --mem-fraction-static 0.91 --attention-backend flashinfer`, `--mamba-ssm-dtype bfloat16`. Measured 2026-09-05.

## Serving throughput, cookbook shape

Same shape SGLang validated its RTX 5090 cookbook cells at: `sglang.bench_serving`, random dataset, input 8192, output 1024, concurrency 1. Random-token prompts vary with the seed, so every row uses `--seed 7 --num-prompts 10 --warmup-requests 1`.

```
python3 -m sglang.bench_serving --backend sglang --port 30000 --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1 \
  --num-prompts 10 --max-concurrency 1 --warmup-requests 1 --seed 7
```

| config | output tok/s | mean TPOT ms | median TPOT ms | accept | step ms (mean TPOT × accept) |
| --- | ---: | ---: | ---: | ---: | ---: |
| cutlass + bf16 draft (stock SGLang) | 173.0 | 4.95 | 3.97 | 4.47 | 22.1 |
| b12x + bf16 draft | 196.2 | 4.31 | 3.46 | 4.25 | 18.3 |
| b12x + FP8 draft | 212.4 | 3.89 | 3.08 | 4.77 | 18.6 |
| b12x + all-NVFP4 target + bf16 draft | 263.9 | 3.12 | 2.92 | 4.36 | 13.6 |
| b12x + all-NVFP4 target + FP8 draft | 241.5 | 3.46 | 2.82 | 4.11 | 14.2 |

Accept moves between runs of this shape even at a fixed seed (the bf16 draft is the same weights in rows 1 and 2, and the GEMM outputs are bit-identical, yet accept reads 4.47 and 4.25). Ten random-token prompts are a small sample. The last column removes accept from the comparison: the backend change cuts the step from 22.1 to 18.3 ms, 17%, and the all-NVFP4 target cuts it again to 13.6 ms, 38% below stock. The FP8 draft rows move with accept on this shape, not with their step.

Six-prompt runs at the default seed, for reference: stock 153.9 tok/s, median TPOT 5.34, accept 3.92; b12x 172.2, 4.67, 3.89; b12x + FP8 draft 170.7, 5.21, 3.71. The SGLang cookbook reports DFlash2 on a 5090 at 4.92 ms median TPOT and accept 4.29 on this shape, so the stock rows sit where their measurement does.

Raw output: `receipts/bench_*.jsonl`.

## Acceptance length on named datasets

Greedy, thinking on, `max_tokens 1024`, one request at a time, chat endpoint. Accept is the mean of the server's per-batch `accept len` over each prompt's decode, then the token-weighted mean across prompts. Throughput is total generated tokens over total decode seconds. The server logs one decode line per 40 steps, so a prompt that finishes in fewer than about 130 generated tokens has no accept sample and is excluded from both columns; the `n` column is the count kept.

| dataset | prompts | kept | tokens | b12x + bf16 draft tok/s | accept | b12x + FP8 draft tok/s | accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MT-Bench (first turn) | 80 | 75 | 50,078 | 190.5 | 3.80 | 201.5 | 3.83 |
| HumanEval | 164 | 164 | 116,519 | 239.2 | 4.64 | 248.8 | 4.58 |
| GSM8K (test, first 100) | 100 | 83 | 29,751 | 264.0 | 5.07 | 277.0 | 5.02 |
| MATH-500 (first 100) | 100 | 88 | 49,405 | 261.6 | 5.14 | 272.4 | 5.08 |

Same sets with the all-NVFP4 target ([thomasgardiner/Qwen3.8-27B-NVFP4-all](https://huggingface.co/thomasgardiner/Qwen3.8-27B-NVFP4-all)), b12x:

| dataset | bf16 draft tok/s | accept | accuracy | FP8 draft tok/s | accept | accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MT-Bench (first turn) | 215.1 | 3.72 | | 232.1 | 3.78 | |
| GSM8K (test, first 100) | 309.2 | 5.06 | 88% | 317.0 | 4.95 | 84% |
| MATH-500 (first 100) | 303.2 | 5.05 | 64% | | | |

Accuracy is `bench/grade.py`, an answer-tail match against the reference at a 1024-token cap: GSM8K read 87% and 90% with the two drafts on the FP8-mixed target and 88% and 84% here, so the spread of one hundred graded items is about ±3 points. It is a drift check, not a leaderboard. The default we serve is the all-NVFP4 target with the bf16 draft.

Token and kept counts are from the bf16 run; the FP8 run kept 75, 162, 82, 88 prompts. Across the four sets the FP8 draft changes accept by +0.8%, −1.3%, −1.0%, −1.2% and tok/s by +5.8%, +4.0%, +4.9%, +4.1%. An earlier five-prompt probe on hand-written code tasks showed an 8% accept drop; 164 HumanEval prompts put it at 1.3%. The larger sample is the one to trust.

Prompts: `bench/stage_datasets.py`. Runner: `bench/dataset_bench.py`. Raw per-prompt rows with output hashes: `receipts/ds_*.jsonl`.

The cutlass backend does not change acceptance; it changes step time only. Its dataset tok/s is the bf16 row scaled by the step ratio from the serving table, about 0.83.
