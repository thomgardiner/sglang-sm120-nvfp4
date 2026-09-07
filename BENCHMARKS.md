# Benchmarks

One RTX 5090, SGLang image `lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe`, target Qwen3.8-27B NVFP4 (ModelOpt), DFlash2 draft, 8 draft tokens, `--max-running-requests 1 --cuda-graph-max-bs-decode 1 --mem-fraction-static 0.91 --attention-backend flashinfer`, `--mamba-ssm-dtype bfloat16`. Measured 2026-09-05.

## Serving throughput, cookbook shape

Same shape SGLang validated its RTX 5090 cookbook cells at: `sglang.bench_serving`, random dataset, input 8192, output 1024, concurrency 1. Random-token prompts vary with the seed, so every row uses `--seed 7 --num-prompts 10 --warmup-requests 1`.

```
python3 -m sglang.bench_serving --backend sglang --port 30000 --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1 \
  --num-prompts 10 --max-concurrency 1 --warmup-requests 1 --seed 7
```

| config | output tok/s | mean TPOT ms | median TPOT ms | accept |
| --- | ---: | ---: | ---: | ---: |
| cutlass + bf16 draft (stock SGLang) | 173.0 | 4.95 | 3.97 | 4.47 |
| b12x + bf16 draft | 196.2 | 4.31 | 3.46 | 4.25 |
| b12x + FP8 draft | 212.4 | 3.89 | 3.08 | 4.77 |
| b12x + all-NVFP4 target + bf16 draft | 263.9 | 3.12 | 2.92 | 4.36 |
| b12x + all-NVFP4 target + FP8 draft | 241.5 | 3.46 | 2.82 | 4.11 |

Accept moves between runs of this shape even at a fixed seed (the bf16 draft is the same weights in rows 1 and 2, and the GEMM outputs are bit-identical, yet accept reads 4.47 and 4.25). Ten random-token prompts are a small sample, and the accept column is the server log's mean over the run, not a per-request value. Do not quote 263.9 tok/s as +52% over 173.0: those two rows are one seed of random tokens, and the 264 run is the shipping config (all-NVFP4, bf16 draft), not the "all three" config in the README stacking table. Step time and per-prompt tok/s are measured directly in the next section; do not derive step time from this table. An earlier version of this table printed a step column as mean TPOT × accept (22.1, 18.3, 18.6, 13.6, 14.2 ms). Those accept samples did not belong to those runs, and the 13.6 figure was wrong by 2.7 ms.

Six-prompt runs at the default seed, for reference: stock 153.9 tok/s, median TPOT 5.34, accept 3.92; b12x 172.2, 4.67, 3.89; b12x + FP8 draft 170.7, 5.21, 3.71. The SGLang cookbook reports DFlash2 on a 5090 at 4.92 ms median TPOT and accept 4.29 on this shape, so the stock rows sit where their measurement does.

Raw output: `receipts/bench_*.jsonl`.

## Decode step and tok/s, measured directly

`bench/step_time.py`: one streamed chat completion at a time, thinking off, greedy, 512 output tokens. SGLang emits one stream chunk per verify step at batch 1, so step time is decode seconds divided by chunks, tokens per step is completion tokens divided by chunks, and tok/s is completion tokens over decode seconds. SGLang's logged accept length counts `num_correct_drafts + 1` per step, so it is the same quantity as tokens per step. Three prompts plus a 9040-token prose prompt, two repeats each, means shown. GPU1, 2026-09-06. Raw output: `receipts/step/step-*.json`.

| config | prose tok/s | code tok/s | math tok/s | step ms | tok/step (prose, code, math) |
| --- | ---: | ---: | ---: | ---: | --- |
| cutlass + RadixArk target + bf16 draft (stock SGLang) | 125.5 | 259.6 | 279.8 | 21.5 | 2.71, 5.57, 6.02 |
| b12x + RadixArk target + bf16 draft | 143.8 | 297.7 | 321.1 | 18.8 | 2.71, 5.57, 6.02 |
| b12x + RadixArk target + FP8 draft | 144.3 | 332.6 | 316.2 | 18.0 | 2.60, 5.95, 5.69 |
| b12x + all-NVFP4 target + bf16 draft | 156.3 | 346.4 | 353.1 | 16.3 | 2.56, 5.63, 5.76 |
| b12x + all-NVFP4 target + FP8 draft | 167.9 | 356.3 | 396.5 | 15.5 | 2.61, 5.50, 6.15 |

Step time does not depend on the prompt: the spread across prose, code, and math is under 0.2 ms, and repeats agree within 0.2 ms. The 9040-token prompt adds 0.3 to 0.4 ms. The b12x backend and the all-NVFP4 target change step time only; tokens per step is identical between the first two rows because the GEMM outputs are bit-identical. The FP8 draft changes the draft's numerics, so tokens per step moves with it, up on code and down on math here, and the tok/s change ranges from 0% to +12% across the three prompts on the RadixArk target.

The step floor is set by bytes: the all-NVFP4 target streams 16.22 GB of weights and scales per step and the bf16 draft 3.85 GB, and an RTX 5090 reads at 1701 GB/s (measured, 94.9% of the 1792 spec), so the fastest bf16-draft row runs at 72% of the memory ceiling.

The same three arms, interleaved, three rounds, one session, n=6 per cell (mean and half-spread). GPU1 idle neighbor at 34 W. Receipts: `receipts/step/step-il-*.json`.

| arm | prose | code | math | prose 9k | step |
| --- | ---: | ---: | ---: | ---: | ---: |
| stock SGLang | 125.3 ± 0.5 | 259.5 ± 0.1 | 279.8 ± 2.1 | 117.6 ± 6.9 | 21.62 ms |
| shipping (b12x + all-NVFP4 + bf16 draft) | 156.6 ± 0.4 | 346.4 ± 0.2 | 353.2 ± 10.7 | 150.3 ± 1.5 | 16.40 ms |
| best (+ FP8 draft) | 168.1 ± 0.4 | 356.4 ± 0.2 | 396.6 ± 19.8 | 164.5 ± 3.5 | 15.60 ms |

Shipping over stock: +24.9%, +33.5%, +26.2%, +27.9%. Best over stock: +34.1%, +37.4%, +41.7%, +39.9%. The sequential n=2 table above agrees with these endpoints within 0.3 tok/s. The README lead table is this shipping column, rounded.

## Draft length

`--speculative-num-draft-tokens` defaults to 8 in the SGLang cookbook cells for this model. It is not the best value for every workload. Same measurement as the previous section, four repeats per prompt, all-NVFP4 target with the bf16 draft. Raw output: `receipts/step/step-dt-*.json` and `step-dtc-*.json`.

| draft tokens | prose tok/s | code tok/s | math tok/s | step ms | tok/step (prose, code, math) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 149.9 | 222.8 | 215.4 | 16.3 | 2.45, 3.63, 3.52 |
| 8 | 156.7 | 346.4 | 353.9 | 16.3 | 2.56, 5.63, 5.76 |
| 12 | 149.1 | 363.8 | 425.9 | 17.0 | 2.55, 6.17, 7.22 |
| 16 | 138.3 | 362.7 | 416.0 | 17.7 | 2.46, 6.40, 7.37 |

Going from 8 to 12 costs 0.7 ms of step time, because the verify checks four more tokens against the same weights, and the weights dominate. Whether that pays depends on whether the extra draft tokens are accepted. On math it is worth +20%, on code +5%, and on prose it is a 5% loss because acceptance does not rise at all there. A 9040-token prompt behaves like prose.

The math rows are bimodal, 405 and 446 tok/s on alternating repeats, reproducibly. The two repeats send the same prompt, so the second reuses the prefix cache; the split is stable across separate server starts.

`--speculative-adaptive` changes nothing here. It adjusts `num_steps` from the acceptance rate, and DFlash2 runs with `--speculative-num-steps 1`, so there is nothing for it to adjust. Measured identical to the 8-token rows on all three prompts.

The default in this repo stays 8, which is the best single value for prose and within 5% on code. Serve math or code workloads with 12.

```
bench/draft_token_sweep.sh
```

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

The cutlass backend does not change acceptance; it changes step time only. Stock CUTLASS was not run on these named datasets. Do not scale the b12x rows by 18.8/21.5 and publish that as a measurement.

## Accuracy, lm-evaluation-harness

Both targets served by SGLang with b12x and the bf16 DFlash2 draft, one target per GPU, thinking disabled through `--default-chat-template-kwargs '{"enable_thinking": false}'`, greedy, 8 concurrent requests, zero-shot chat. `max_gen_toks` 4096, 2048 for IFEval and HumanEval. Output in `receipts/lmeval/nothink-*/`.

| Benchmark | Metric | RadixArk/Qwen3.8-27B-NVFP4 | Qwen3.8-27B-NVFP4-all |
| --- | --- | ---: | ---: |
| GSM8K (1319) | exact match, flexible-extract | 85.90 ± 0.96 | 85.82 ± 0.96 |
| MATH-500 | math_verify | 86.00 ± 1.55 | 84.80 ± 1.61 |
| GPQA Diamond (198) | CoT zero-shot, exact match | 68.18 ± 3.32 | 67.68 ± 3.33 |
| IFEval (541) | prompt-level strict | 79.30 ± 1.74 | 80.96 ± 1.69 |
| IFEval (541) | prompt-level loose | 83.55 ± 1.60 | 84.47 ± 1.56 |
| HumanEval (164) | pass@1, executed | 93.29 (153/164) | 92.68 (152/164) |

Every difference is inside one standard error. Two harness traps, both visible in the receipts:

- With thinking on, the model spends its budget in `reasoning_content`, which the harness never reads. MATH-500 scored 49% with a median answer length of 29 characters. Thinking off is the only honest way to run this model through `local-chat-completions`.
- `humaneval_instruct` assumes the prompt ends inside an open code fence and stops on `\ndef`. A chat model opens its own fence, so the built-in filter scores 0. `bench/he_grade.py` takes the longest fenced block from each sample, appends the task's `test` and `check(entry_point)`, and runs it with a 15 s timeout. `humaneval_executed.json` holds the failed task ids.

Earlier thinking-on run, GSM8K first 500, one request at a time: 83.0 ± 1.7 (RadixArk) and 85.2 ± 1.6 (all-NVFP4).

```
bench/eval_suite.sh 0 nvfp4all     # GPU 0: stop the service, serve the eval spec, run five tasks, grade HumanEval
bench/eval_suite.sh 1 radixark     # GPU 1, same, other target
```
