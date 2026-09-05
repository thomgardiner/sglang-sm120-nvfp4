# SGLang NVFP4 on RTX 5090: use FlashInfer's SM120 kernel

SGLang picks the `flashinfer_cutlass` NVFP4 GEMM on SM120 (RTX 5090). FlashInfer ships a faster SM120 kernel, `b12x`, that SGLang never selects. A two-file patch adds the choice and makes SM120 auto pick it.

Result on Qwen3.8-27B-NVFP4 with DFlash2 speculative decoding, thinking on, one stream, one RTX 5090: **+15% decode tok/s, bit-identical output.**

| prompt (greedy, 1200 tokens) | cutlass tok/s | b12x tok/s |
| --- | ---: | ---: |
| prose | 149.5 | 171.5 |
| math | 230.1 | 264.6 |
| Rust code | 191.7 | 220.2 |

Output SHA-256 of the generated text matched across backends and repeats. Receipts: `receipts/greedy-gpu1-*.json`.

## Why

Cold-weight GEMM at M=9 (8 weight copies rotated to defeat L2), percent of 1.792 TB/s:

| shape | NVFP4 `cutlass` | NVFP4 `b12x` |
| --- | ---: | ---: |
| gate_up N=34816 K=5120 | 63% | 89% |
| down N=5120 K=17408 | 52% | 78% |
| lm_head N=248320 K=5120 | 65% | 92% |

`bench/gemm_cold.py`, `bench/fp4_auto.py`. In the live trace the NVFP4 GEMMs were 10.5 ms of a 23.4 ms decode step. After the patch: 7.6 ms.

## Apply

```
cd sglang && git apply /path/to/b12x.patch
python3 -m sglang.launch_server ... --fp4-gemm-backend flashinfer_b12x
```

Or leave `--fp4-gemm-backend auto`; the patch makes SM120 pick `b12x`.

Tested on image `lmsysorg/sglang@sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe`, FlashInfer 0.6.17, CUDA 13.0, driver for 2× RTX 5090.

## Reproduce

Server: `Qwen/Qwen3.8-27B` NVFP4 checkpoint, `--speculative-algorithm DFLASH --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8 --max-running-requests 1 --cuda-graph-max-bs-decode 1 --mem-fraction-static 0.91 --attention-backend flashinfer`.

```
export SGLANG_API_KEY=...          # or leave empty
python3 bench/greedy_fixed.py 30000 out.json
```

Run once with `--fp4-gemm-backend flashinfer_cutlass`, once with `flashinfer_b12x`. Compare `decode_tok_s` and `sha`.

`bench/profile_step.sh 30000 <container>` records 40 scheduler steps with SGLang's torch profiler; `bench/trace_steps.py <trace.gz>` prints per-step kernel time.

## License

MIT. The patch applies to SGLang, which is Apache-2.0.
