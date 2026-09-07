# Qwen3.8-27B NVFP4 + DFlash2 on an RTX 5090: +25 to +33% tok/s

Shipping config on this card: FlashInfer `b12x`, [all-NVFP4 target](https://huggingface.co/thomasgardiner/Qwen3.8-27B-NVFP4-all), bf16 DFlash2 draft, 8 draft tokens. Versus stock SGLang (CUTLASS NVFP4, RadixArk target, same draft) on the same GPU, greedy, thinking off, three rounds interleaved, n=6 per cell:

| prompt | stock SGLang | shipping | change |
| --- | ---: | ---: | ---: |
| prose | 125 tok/s | 157 tok/s | +25% |
| code | 259 tok/s | 346 tok/s | +33% |
| math | 280 tok/s | 353 tok/s | +26% |
| prose, 9k context | 118 tok/s | 150 tok/s | +28% |

Decode step 21.6 → 16.4 ms, a 24% shorter step, independent of the prompt. Adding the [FP8 draft](https://huggingface.co/thomasgardiner/Qwen3.8-27B-DFlash2-FP8) on top of shipping is 168 / 356 / 397 tok/s (+34% / +37% / +42% over stock) and 15.6 ms per step. That increment moves acceptance, so it is prompt-dependent.

Each change on its own, same three prompts. Stock, shipping, and FP8-draft endpoints are the interleaved means above. The two middle rows are an earlier sequential run (n=2) that matches those endpoints:

| change | what | prose | code | math | step |
| --- | --- | ---: | ---: | ---: | ---: |
| stock SGLang, RadixArk export, bf16 draft | | 125 | 259 | 280 | 21.6 ms |
| [`b12x.patch`](b12x.patch) | SGLang picks FlashInfer's SM120 NVFP4 kernel instead of CUTLASS | 144 | 298 | 321 | 18.8 ms |
| [Qwen3.8-27B-NVFP4-all](https://huggingface.co/thomasgardiner/Qwen3.8-27B-NVFP4-all) | GDN and attention projections in NVFP4 instead of FP8 | 157 | 346 | 353 | 16.4 ms |
| [Qwen3.8-27B-DFlash2-FP8](https://huggingface.co/thomasgardiner/Qwen3.8-27B-DFlash2-FP8) | draft MLP and o_proj in FP8 | 168 | 356 | 397 | 15.6 ms |

The first two changes leave acceptance alone. The FP8 draft does not. Full tables, commands, and raw logs: [BENCHMARKS.md](BENCHMARKS.md) and [receipts/](receipts/). Checkpoints: [Hugging Face collection](https://huggingface.co/collections/thomasgardiner/qwen38-27b-on-rtx-5090-6a9cd2afdbba0fcd6e11a711).

One `bench_serving` run on random-token prompts (cookbook shape, seed 7) read 173 → 264 tok/s on the shipping config. That is a different measurement from the table, and acceptance moved. Do not quote it. The row is labelled in BENCHMARKS.md.

## The patch

On SM120 (RTX 5090, RTX PRO 6000) SGLang's `--fp4-gemm-backend auto` resolves to `flashinfer_cutlass`. FlashInfer has a kernel written for SM120, `b12x`, and prefers it in its own auto mode. SGLang has no option that selects it. The patch adds `flashinfer_b12x` and makes `auto` pick it on SM120.

Cold GEMM at M=9, percent of HBM bandwidth:

| shape | cutlass | b12x |
| --- | ---: | ---: |
| gate_up 34816×5120 | 63% | 89% |
| down 5120×17408 | 52% | 78% |
| lm_head 248320×5120 | 65% | 92% |

Output is bit-identical. Same prompts, greedy, 1200 tokens, same GPU:

| prompt | cutlass | b12x |
| --- | ---: | ---: |
| prose | 149.5 | 171.5 |
| math | 230.1 | 264.6 |
| code | 191.7 | 220.2 |

```
cd sglang && git apply /path/to/b12x.patch
```

Upstream: [sgl-project/sglang#38170](https://github.com/sgl-project/sglang/pull/38170). Needs CUDA 13 and NVFP4; FlashInfer 0.6.18 enables b12x on SM120 and SM121.

## The two checkpoints

The target export from RadixArk leaves 6.7 GB of GDN and attention projections in FP8. `bench/nvfp4_convert.py` re-quantizes them to NVFP4 from the bf16 source, reusing the export's activation calibration. GSM8K, MATH-500, GPQA Diamond, IFEval, and HumanEval all land within one standard error of the RadixArk export (lm-evaluation-harness, thinking off), acceptance within 2%.

The DFlash2 draft is 3.85 GB of bf16. `bench/quant_draft.py` puts its MLP and o_proj in FP8 and leaves q/k/v alone so SGLang's fused KV path stays on. Acceptance within 1.3% on MT-Bench, HumanEval, GSM8K, MATH-500.

## Reproduce

```
export SGLANG_API_KEY=...
python3 bench/greedy_fixed.py 30000 out.json           # fixed prompts, tok/s and output hash
python3 bench/stage_datasets.py                        # MT-Bench, HumanEval, GSM8K, MATH-500 prompts
python3 bench/dataset_bench.py 30000 <container> gsm8k out.jsonl
python3 bench/grade.py gsm8k out.jsonl
python3 -m sglang.bench_serving --backend sglang --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 --num-prompts 10 --max-concurrency 1 --seed 7
```

Apache-2.0.
