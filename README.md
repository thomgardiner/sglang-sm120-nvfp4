# SGLang NVFP4 on RTX 5090: select FlashInfer's SM120 kernel

SGLang runs NVFP4 linear layers on SM120 (RTX 5090, RTX PRO 6000) through the `flashinfer_cutlass` GEMM. FlashInfer also ships an SM120-specific NVFP4 kernel, `b12x`, and prefers it in its own auto selection, but SGLang has no option that selects it. This patch adds the `flashinfer_b12x` choice and makes SGLang's `auto` pick it on SM120.

Measured on Qwen3.8-27B NVFP4 with DFlash2 speculative decoding, thinking on, one stream, one RTX 5090: **15% more decode tokens per second, same output.**

| prompt, greedy, 1200 tokens | cutlass tok/s | b12x tok/s |
| --- | ---: | ---: |
| prose | 149.5 | 171.5 |
| math | 230.1 | 264.6 |
| Rust code | 191.7 | 220.2 |

The SHA-256 of the generated text is equal across both backends and both repeats. Receipts: `receipts/`.

## Why it is faster

Cold-weight GEMM at M=9, weights rotated through 8 copies to defeat the 128 MB L2, percent of 1.792 TB/s:

| shape | `cutlass` | `b12x` |
| --- | ---: | ---: |
| gate_up N=34816 K=5120 | 63% | 89% |
| down N=5120 K=17408 | 52% | 78% |
| lm_head N=248320 K=5120 | 65% | 92% |

In the live decode step the NVFP4 GEMMs took 10.5 ms of 23.4 ms. With b12x they take 7.6 ms. Scripts: `bench/gemm_cold.py`, `bench/fp4_auto.py`.

## Apply

```
cd sglang
git apply /path/to/b12x.patch
```

Then start the server as before. `--fp4-gemm-backend auto` now selects b12x on SM120. You can also pass `--fp4-gemm-backend flashinfer_b12x`.

The patch is against SGLang `main` at `77aee202` (2026-09-05). The measurements used the image `lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe`, where the same two-line change applies.

## FP8 draft, a second 4%

The DFlash2 draft reads 3.85 GB of bf16 weights per step. [thomasgardiner/Qwen3.8-27B-DFlash2-FP8](https://huggingface.co/thomasgardiner/Qwen3.8-27B-DFlash2-FP8) stores the draft's MLP and o_proj tensors in FP8 with one scale per tensor and leaves q, k, v in bf16 so SGLang keeps its fused DFlash KV path. Step time from the server log on the same greedy prompts, on top of b12x:

| draft | mean accept | step (ms) |
| --- | ---: | ---: |
| bf16 | 4.01 | 19.10 |
| FP8 per-channel | 3.87 | 19.37 |
| FP8 per-tensor | 3.99 | 18.31 |

Per-channel scales route to a CUTLASS FP8 GEMM at 25 to 60% of bandwidth on SM120. A per-tensor scale routes to cuBLAS at 86%. Quantizing q, k, v turns the fused KV path off and is slower. `bench/quant_draft.py` builds the checkpoint; receipts in `receipts/`.

Point `--speculative-draft-model-path` at the Hugging Face repo and do not pass `--speculative-draft-model-quantization`.

## Limits

FlashInfer's b12x requires CUDA 13 or later and NVFP4 with the 128x4 scale layout. It does not cover MXFP4. FlashInfer excludes SM121 (GB10, DGX Spark) from b12x on purpose, so this patch changes nothing there.

## Reproduce

Server flags used: `--speculative-algorithm DFLASH --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8 --max-running-requests 1 --cuda-graph-max-bs-decode 1 --mem-fraction-static 0.91 --attention-backend flashinfer`, target checkpoint Qwen3.8-27B NVFP4 (ModelOpt export).

```
export SGLANG_API_KEY=...     # omit if the server has no key
python3 bench/greedy_fixed.py 30000 out.json
```

Run once with `--fp4-gemm-backend flashinfer_cutlass` and once with `flashinfer_b12x`. Compare `decode_tok_s` and `sha` in the two files.

To see the per-kernel step budget: `bench/profile_step.sh <port> <container>` records 40 scheduler steps with SGLang's built-in profiler, then `bench/trace_steps.py <trace.json.gz>` prints kernel time per decode step.

## License

Apache-2.0, the same license as SGLang.
