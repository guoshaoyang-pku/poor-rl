# Precision recipes

RL fine-tuning updates are tiny (lr ~1e-6..1e-5). In bf16, a weight w gets an
update u only if |u| is larger than roughly 2^-8 * |w| (bf16 has 8 mantissa bits);
smaller updates are silently rounded away. At lr=2e-6 that rounding floor is a
real fraction of the intended update, so **pure bf16 training can eat learning
entirely** -- we observed exactly this in our precision arms before settling on
the recipes below.

## The three recipes (launcher env: `DTYPE` x `MIXED_PRECISION`)

| Recipe | env | master weights | compute | when |
|---|---|---|---|---|
| pure fp32 | `DTYPE=none MIXED_PRECISION=no` | fp32 | fp32 | debugging; arm A reference |
| pure bf16 | `DTYPE=bfloat16 MIXED_PRECISION=no` | bf16 | bf16 | fast default at 0.8B with lr >= 2e-6 |
| fp32 master + bf16 compute | `DTYPE=none MIXED_PRECISION=bf16` | fp32 | bf16 (autocast) | **the low-precision recipe**: small lr, long runs, or whenever updates approach the bf16 rounding floor |

The third recipe is the one to build on for bf16-and-lower work: optimizer states
and master weights stay fp32 (accelerate keeps the model in fp32 and autocasts the
forward), so a 2e-6 update always lands, while the matmuls run at bf16 speed.

## Going lower than bf16

- FP8 rollout is independent of actor precision. The launcher defaults to `KV_DTYPE=fp8` on the validated H200 setup; set `KV_DTYPE=auto` to disable it or compare against BF16/auto. It can reduce KV-cache memory substantially. On other accelerators, verify vLLM support before use.
- Native CPU KV offload is a recommended serving SOP when long contexts or high concurrency need additional KV capacity. It passed a 512-request paired test with a 16-GiB CPU tier on vLLM 0.30 / H200 using FP8 KV; treat the result as workload-specific and revalidate throughput, request success, and host-memory pressure on the target backend.
- CPU offload of actor/model parameters is separate from KV offload: FSDP2 parameter offload is implemented but remains disabled by default because measured training steps were about 64% slower. Use only when its memory savings are needed.
- `ROLLOUT_QUANTIZATION=fp8` asks vLLM to quantize rollout weights/compute. It can save additional memory and is now measured on identical token ids against a bf16-weight control (see "Logprob-gap probe" below): FP8 multiplies per-token logprob noise ~5x while moving the **sequence-level** GSPO ratio by only ~1-3%. It pairs correctly with an fp32 LoRA adapter (`docs/LORA.md`). FP8 KV stays the validated default; FP8 weights remain an opt-in per-experiment choice, since long-run quality is still unproven.
- Keep actor master weights FP32 (`DTYPE=none`) with BF16 autocast (`MIXED_PRECISION=bf16`) for this experiment. Do not raise the learning rate merely to compensate for FP8 inference; first check policy-ratio/clip health against a BF16 rollout control.
- A high `gspo/seq_clip_low_frac` is **not** a precision symptom, and widening eps is not the fix. Measured cause on Qwen3.5-0.8B: the async trainer packs a rank's sequences into one forward with per-sequence `position_ids`, which does not reset the hybrid GatedDeltaNet layers' conv/recurrent state, so packed logprobs drift by -0.10 to -0.62/token depending on pack position. BF16, FP8-weight and FP8-KV arms all show it; diagnosis and fix options are in `docs/LORA.md` ("Do not pack a hybrid model's sequences").

### H200 smoke and paired held-out validation (vLLM 0.30, Qwen3.5-0.8B, 2026-10-01)

- With the same 24,576-token max model length, 0.85 GPU-memory utilization, eager mode, and single-GPU serving, `KV_DTYPE=fp8` reported 15,949,824 cached tokens (649.0 max-length sequences) vs 8,764,179 tokens (356.6 sequences) for BF16/auto KV: **1.82x cache capacity**. Both modes loaded and completed generation requests.
- Paired evaluation used the same `async_dp_flip450` checkpoint-500, 50 held-out questions, two samples per question, T=1.0/top-p=1.0/max-tokens=16,384, 16 concurrent requests, and matched request seeds. Across three repeated passes, mean accuracy was 33.7% (auto) vs 35.0% (FP8); parse rate 95.7% vs 95.3%; truncation 0% in both. Median generated-token throughput was 284 vs 309 tokens/s (**+8.6%** FP8), while mean request latency was 1.54 vs 1.46 s. This is a small, single-model/task probe: no statistically reliable accuracy gain is claimed, and other prompt lengths/load levels may show different throughput.
- FP8-weight GSPO was run for 3 steps on 360-1 H200 (vLLM 0.30, Qwen3.5-0.8B), with FP32 actor/master weights, BF16 autocast/compute, FP8 rollout weights, FP8 KV cache, 2e-6 LR, and matched 3-step BF16-weight control. Both completed live weight sync each step and produced changed FP32 checkpoints; nonzero gradients were observed (FP8 norm 9.1–16.1, BF16 norm 6.6–12.4). The brief run proves the training/sync/update path executes, **not** learning quality. Both arms showed very high `gspo/seq_clip_low_frac` (~0.98–1.00), so diagnose ratio/logprob alignment on a representative run before scaling up; FP8 weights are not approved as a production default.
- FP8 KV cache is marked **operationally validated and default for H200 runs**. On unsupported accelerators or when investigating regressions, explicitly set `KV_DTYPE=auto`. Temporary vLLM services were stopped after testing; the only remaining GPU usage was an unrelated user-owned benchmark, left untouched.

### Logprob-gap probe (2026-10-03, 360-1 H200, vLLM 0.30)

Same token ids scored by the vLLM server under three precision configs and by the
trainer-side model (`delta = logp_trainer - logp_rollout`), 5 sequences / 1044
continuation tokens:

| rollout config | per-token abs_mean | sequence-level abs_mean | sequence rho range |
|---|---|---|---|
| BF16 weights + BF16 KV | 0.0104 | 0.0019 | 0.9955 - 1.0013 |
| FP8 weights + BF16 KV | 0.0482 | 0.0087 | 0.9923 - 1.0271 |
| FP8 weights + FP8 KV | 0.0533 | 0.0091 | 0.9936 - 1.0230 |

FP8 weights and FP8 KV are close to additive, and the quantity GSPO actually clips
on (sequence ratio) stays within ~1-3% of 1.0. For reference, vLLM's own
generation-time vs prefill-scoring logprobs differ by -0.016/token (rho 0.984) on
the same server and ids, so the plumbing noise floor is already comparable to the
entire FP8 contribution. The probe was run with a trained fp32 LoRA adapter
(`docs/LORA.md`), which contributes ~1e-3 of the gap by itself.

## FP4 on H200: what works, what does not (2026-10-03)

Measured on 360-1 H200 (sm90). The headline is that **FP4 rollout is possible on
Hopper, but only via MXFP4 — and only as weight-only A16.**

| path | works on sm90? | reason |
|---|---|---|
| `fp_quant` | **no** | `get_min_capability() == 100`; hard Blackwell gate |
| `nvfp4_per_token` | **no, and a no-op** | the online shorthand sets only `moe=QuantSpec(...)`; `linear` is left unquantized. The MoE method raises `ValueError` unless `is_device_capability_family(100)`. **On a dense model there are no MoE layers, so it quantizes nothing at all — which is why it starts with no error.** |
| `modelopt_fp4` | **yes for W4A16, needs a checkpoint** | the *checkpoint* is what is missing, not the hardware: there is no `Nvfp4OnlineLinearMethod`, so NVFP4 cannot be produced at load time, and `nvidia-modelopt` is not installed. With an NVFP4 checkpoint this resolves to `MarlinNvFp4LinearKernel`, which is supported here — see below. |
| **online `mxfp4`** | **yes** | `Mxfp4OnlineLinearMethod` has **no arch gate**; it resolves to `MarlinMxFp4LinearKernel` (`is_fp4_marlin_supported()` = `is_cuda() and capability >= 75`) |

### Correction: NVFP4 does *not* require a Blackwell card

Two different things are easy to conflate, and an earlier revision of this note
conflated them:

- **Native FP4 tensor-core compute** requires Blackwell. In vLLM that is the
  `pytorch` kernel (`is_device_capability_family(100)`), `flashinfer`
  (`sm_100` / `sm_12x`), and `fp_quant`.
- **NVFP4 weight-only (W4A16)** runs on **Hopper**. vLLM ships
  `MarlinNvFp4LinearKernel`, gated only on
  `is_fp4_marlin_supported()` = `is_cuda() and capability >= 75`.

Measured on 360-1 (sm90): `is_fp4_marlin_supported()` returns **True**, and the
kernel self-describes when used:

> Your GPU does not have native support for FP4 computation but FP4 quantization
> is being used. Weight-only FP4 compression will be used leveraging the Marlin
> kernel. This may degrade performance for compute-heavy workloads.

This is also how QeRL can claim "RL for 32B LLMs on a single H100 GPU" while using
NVFP4: on Hopper the gain is **memory**, not 4-bit tensor-core throughput. That is
consistent with our own measurement that weight format does not move decode
throughput (bf16 -> MXFP4 = -1.5%).

What actually blocks NVFP4 *for us* is the **checkpoint**, not the card: there is
no `Nvfp4OnlineLinearMethod` (the online linear methods are fp8 x3, mxfp4, mxfp8
only), and `nvidia-modelopt` is not installed, so we cannot produce one offline.
Online MXFP4 is therefore the practical stand-in: same W4A16 memory win, differing
only in scale format (E8M0 / block-32 vs E4M3 / block-16).

### Multimodal checkpoints: exclude the vision tower

MXFP4 quantizes in block-32 groups, so the input dim of every quantized 2-D weight
must be divisible by 32. On Qwen3.8-27B the vision tower violates this: **27
weights** of the form `model.visual.blocks.*.mlp.linear_fc2.weight` have shape
`[1152, 4304]`, and 4304 % 32 = 16. Without an exclusion the server dies at init:

```
ValueError: MXFP4 requires input_size_per_partition (4304) to be divisible by 32.
```

Fix (online `ignore` accepts fnmatch patterns):

```bash
--quantization online \
--quantization-config '{"linear": "mxfp4", "ignore": ["*visual*"]}'
```

The language backbone is unaffected, and that is the part RL trains.

Working invocation (dense model):

```bash
vllm serve "$MODEL" --dtype bfloat16 \
  --quantization online --quantization-config '{"linear": "mxfp4"}'
```

`--quantization mxfp4` **alone does not work**: `mxfp4` is in
`_DEFERRED_ONLINE_SHORTHANDS`, so with no checkpoint metadata
`resolve_quantization_config` returns `None` and the model loads unquantized.

Two caveats the server logs about Marlin MXFP4, both material:
`MarlinMxFp4LinearKernel is a weight-only (A16) kernel; the requested activation
quantization is ignored` (activations stay 16-bit, so there is **no** activation-side
FP4 speedup), and `Marlin requires thread-tile padding for some weight shapes in
this model ... performance may be degraded`.

### Serving matrix (1 GPU, max-len 4096, 256 concurrency, 1024 in / 256 out)

| weights | KV | LoRA | out tok/s @256 | KV tokens | model load |
|---|---|---|---:|---:|---:|
| bf16 | bf16 | no | **10,957** | 2,437,412 | 1.72 GiB |
| fp8 | bf16 | no | 10,932 | 2,463,744 | 1.18 GiB |
| **mxfp4 (Marlin)** | bf16 | no | 10,798 | 2,477,494 | **0.91 GiB** |
| bf16 | **fp8** | no | 10,227 | **3,463,577** | 1.72 GiB |
| mxfp4 | **fp8** | no | 9,881 | **3,520,102** | 0.91 GiB |
| mxfp4 | fp8 | **yes** | **7,954** | 3,483,238 | 1.23 GiB |

Three conclusions:

1. **Weight precision is not a rollout throughput lever.** bf16 -> MXFP4 is
   **-1.5%**; bf16 -> fp8 is -0.2%. Do not expect a weight-format change to buy
   decode speed at this scale.
2. **FP8 KV buys +42% KV capacity** (2.44M -> 3.46M tokens) but **-6% throughput**
   when the run is not capacity-bound (256 concurrency vs a 595-sequence ceiling).
   Its benefit only appears once concurrency actually saturates the cache. This
   also means the earlier "fp8 weights + fp8 KV = 2.82x" reading was **almost
   entirely the KV half**, measured in a capacity-bound regime.
3. **LoRA costs ~15% on rollout** (9,376 -> 7,954 tok/s at 256; p95 6.92 -> 8.17 s).
   MXFP4 + LoRA is nonetheless functional: the kernel stays
   `MarlinMxFp4LinearKernel`, the adapter registers as a servable model, and a
   synthetic scale-0.05 adapter changed all 3/3 test prompts (sum-logprob
   -16.06 -> -41.78 etc.), so the adapter is genuinely applied on top of the
   quantized base.

### 27B measurement: FP4 buys memory, it does not buy speed

Measured on 360-2 (8xH200), Qwen3.8-27B, one GPU per config, `--max-model-len 8192`,
`--gpu-memory-utilization 0.80`, `--kv-cache-dtype fp8`, 1024 in / 256 out:

| metric | bf16 | MXFP4 | delta |
|---|---:|---:|---:|
| model load | 51.1 GiB | **18.29 GiB** | **-32.8 GiB (2.79x smaller)** |
| KV memory | 58.57 GiB | **89.73 GiB** | **+53.2%** |
| KV tokens | 835,584 | **1,280,000** | **+53.2%** |
| max concurrency @8k | 102.0x | **156.25x** | **+53.2%** |
| throughput @256 conc | **1,545.0 tok/s** | 941.0 tok/s | **0.61x** |

| concurrency | 64 | 128 | 256 | 512 | 768 |
|---|---:|---:|---:|---:|---:|
| bf16 | 1,448.5 | 1,443.2 | **1,545.0** | 1,417.2 | 1,359.3 |
| MXFP4 | **981.4** | 928.0 | 941.0 | 880.4 | 839.5 |

**The +53% KV capacity never converts into throughput.** Even at 768 concurrency --
past bf16's own capacity ceiling -- bf16 still wins 1,359 vs 839 tok/s. bf16 peaks at
256 concurrency; MXFP4 peaks at the lowest concurrency tested.

This is exactly what QeRL's claim means, and it is worth stating precisely: QeRL says
a 32B model **fits on one H100**, which is a *feasibility* claim, not a *speed* claim.
On Hopper, **FP4 converts compute into memory**. If the binding constraint is "the
model does not fit", FP4 is the answer. If the binding constraint is rollout
throughput -- which is our case, by a 1.54x deficit -- FP4 is the wrong trade.

Only on Blackwell (native FP4 tensor cores via the `sm_100`/`sm_12x` kernels) could
FP4 plausibly be both faster and smaller. These H200s cannot reach that path.

### Operational gotcha: no nvcc, so disable the flashinfer sampler

Neither 360-1 nor 360-2 has `nvcc` (`which nvcc` is empty, `/usr/local/cuda` does not
exist). vLLM's default sampler JIT-compiles `top_k_top_p_sampling_from_logits` through
`flashinfer`, so a 27B server dies during EngineCore init with:

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

Fix: `export VLLM_USE_FLASHINFER_SAMPLER=0` (uses the native PyTorch sampler). This is
not 27B-specific and not quantization-specific -- it is a property of these images.
Set it on both arms of any comparison so the sampler is held constant.

### Why FP4 training is still not a thing here

The trainer bottleneck is **not** GEMM. Profiler on a controlled bench: the GPU is
saturated 620/629 ms = **98.6%**, but it is eaten by
`vectorized_elementwise_kernel` + `elementwise_kernel` (~8.5 ms each x 37), while
the actual GEMM kernels (`sm90_xmma_gemm`) total only **~92 ms ≈ 15%** of the step.
FP8 kernel calls measured **812 ms vs 121 ms for bf16 (6.7x slower)**.

Controlled trainer throughput (8192 tokens, fused-CE, no fp32-logits artifact):

| config | tok/s | vs bf16 full FT |
|---|---:|---:|
| bf16 full FT | 41,715 | 1.00x |
| bf16 base + LoRA | 30,983 | 0.74x |
| fp8 base + LoRA | 22,001 | 0.53x |

So on Hopper, **both FP8 and FP4 are dead ends on the trainer side**: there are no
FP4 tensor cores, `transformer_engine` is absent, and the GEMM that a lower
precision could accelerate is only 15% of the step. Keep FP32 master + BF16 compute.

**FP4 pays off only where it buys memory for KV capacity/concurrency** — i.e. at
27B (bf16 54 GB -> MXFP4 ~13.5 GB frees ~40 GB), not at 0.8B (0.81 GiB saved,
+1.6% KV tokens, -1.5% throughput).

## How to verify updates are not being eaten

1. Log `grad_norm` (already in the trainer logs) and lr; the expected update RMS
   is ~ lr * grad_norm / sqrt(n_params) per step.
2. Compare against the dtype rounding floor: fp32 ~ 1e-7 relative, bf16 ~ 4e-3
   relative. If expected relative update < floor, switch masters to fp32.
3. Cheap empirical check: two 20-step smokes, pure bf16 vs fp32-master, same seed;
   if the pure-bf16 reward curve lags measurably, updates are being rounded away.
