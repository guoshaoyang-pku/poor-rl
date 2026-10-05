# Recipe: native FP8 training and rollout (experimental)

H200 / SM90, dense Qwen3.5-0.8B, rollout TP=1. This implements FP8 full-parameter
GEMMs with FP32 master weights, gradient accumulation and Adam states. The BF16
production recipe remains the default. The decode alignment gate still fails;
short throughput measurements do not establish long-run learning quality.

Tested environment: PyTorch 2.13.0+cu130, vLLM 0.30.0, Transformers 5.17.0.

## Precision and implementation

| Part | Precision / implementation |
|---|---|
| Master weights, `.grad`, Adam moments | FP32; normal parameter objects and checkpoint keys preserved |
| Linear and LM head forward | E4M3; per-token activation / per-output-channel weight scales; shared native CUTLASS GEMM |
| Linear and LM head backward | E5M2 gradient operands, E4M3 weight/activation operands; native `torch._scaled_mm` with FP32 output and `use_fast_accum=False` |
| Weight updates and rollout reload | Quantize directly from FP32 masters; versioned trainer caches; serving refreshes weight/scales in place, including tied LM head |
| Sensitive operations | BF16/FP32 attention, norms, loss/reductions and recurrent state; GDN `in_proj_a/b`, vision and unaligned projections excluded |
| MLP | Fused gate/up projection, enabled by default (`RLFORGE_FP8_FUSE_MLP=1`) |
| Trainer CUDA graphs | Stateless copies, fresh outputs, bounded cache; Linear and head graph replay both **off by default** |
| Aligned convolution | Optional `RLFORGE_FP8_ALIGN_COMPILE=1`; compile BF16-rounded tap products/sum, retain eager SiLU; **off by default** |
| Aligned pointwise operations | Optional `RLFORGE_FP8_ALIGN_POINTWISE=1`; compile norm epilogues and rounded Q/K division; retain eager reductions and nonlinearities; **off by default** |
| Compact norm backward | Optional `RLFORGE_FP8_ALIGN_BACKWARD=1`; save BF16 inputs and small FP32 norms, recompute and compile RMS/gated RMS/Q/K gradients; reuse the same forward; **off by default** |
| Variable token lengths | Runtime Triton token dimensions; reuse quantization kernels across lengths, with unchanged FP8 bytes/scales |
| Optimizer cache invalidation | Post-step hook advances master versions even for fused AdamW, refreshing Linear, fused gate/up and LM-head caches |

Forward quantization is stateless: `scale = max(amax / 448, 1 / (448 × 512))`,
with true division before E4M3 rounding. Weight and activation bytes/scales were
checked against vLLM PTPC. Trainer and rollout use the same forward quantization
and GEMM, while the surrounding model operations still require numerical gates.

Code: [`fp8.py`](../src/rlforge/fp8.py), [`fp8_serving.py`](../src/rlforge/fp8_serving.py),
[`fast_logprob.py`](../src/rlforge/fast_logprob.py),
[`fp8_alignment.py`](../src/rlforge/fp8_alignment.py).

## Launch and alignment

Install this checkout in both trainer and serving environments so vLLM discovers
the `poor_rl_fp8` plugin. With `ROOT`, `VENV`, `MODEL` and `DATA` set for the
project, a bounded functional run is:

```bash
FP8=native FP8_ALIGN=vllm TP=1 \
SERVER_GPUS=0 TRAINER_GPUS=1 NUM_TRAINER=1 \
NGEN=4 CPS=4 INFLIGHT=8 STALE=1 EPOCHS=1 \
MAX_STEPS=3 MAX_COMPLETION=32 \
RLFORGE_FP8_GRAPHS=0 RLFORGE_FP8_HEAD_GRAPH=0 \
bash scripts/run_async_dp.sh full _fp8_smoke
```

`FP8=native` selects FP32 masters, BF16 autocast, BF16/auto KV, shared-prefix
FP8 logprob and `--quantization poor_rl_fp8`. It currently requires GSPO and
full FT, using `adamw_torch` or `adamw_torch_fused` for FP32 Adam states. The
trainer installs an optimizer post-step hook: fused AdamW can update weights
without advancing their PyTorch version, which would otherwise leave stale
FP8 caches. Standalone users of `fp8.install` must also call
`fp8.register_optimizer_cache_hook(model, optimizer)`.
The launcher also sets `VLLM_DISABLE_COMPILE_CACHE=1`. In vLLM 0.30.0, AOT
cache validation can miss a plugin constructor's embedding dtype change and
reuse a BF16 kernel for an FP32 master. Set this variable **before starting
vLLM** for standalone serving too. Compilation and CUDA graphs remain enabled;
only reuse of the disk compilation cache is disabled.
`FP8_ALIGN=vllm` adds Qwen3.5 BF16 rounding/GDN changes and selects
`--additional-config '{"gdn_prefill_backend":"triton"}'` on rollout. Keep the
same GSPO clip settings as the matched control; alignment is experimental.

## Earlier fixed-batch gates (2026-10-05)

One H200, same checkpoint and 32 fixed historical completions, shared-prefix
forward/backward with activation checkpointing; 24,514 completion tokens.
Checkpoint: `v3_1e_ckpt100_20261003`.
Optimizer, rollout and judge are excluded; head graphs were off in every arm.
FP8 MLP gate/up fusion was on.

| Path / configuration | Median forward/backward ms | Completion token/s | Peak allocated GiB |
|---|---:|---:|---:|
| BF16, elementwise fusion off | 1,119.34 | 21,900 | 91.61 |
| FP8, alignment off, Linear graphs 4 GiB, elementwise fusion off | 1,117.89 | 21,929 | 93.52 |
| BF16, v3.2 elementwise fusion on | 840.59 | 29,163 | 65.39 |
| FP8, alignment on, eager, elementwise fusion on | 1,599.55 | 15,326 | 108.75 |
| FP8, alignment on, compiled convolution, graphs off | 1,111.28 | 22,059 | Separate timing scope |

The unaligned graph arm reaches parity (1.001×) with its BF16 control. The
aligned eager configuration is 1.90× slower than the fused BF16 reference;
this includes its different norm/convolution rounding rules, so it does not
isolate the cost of FP8 GEMMs. These are **not total RL throughput** numbers.
An alternating eager/compiled repeat of the aligned FP8 batch measured
1,593.19 / 1,595.75 ms eager versus 1,111.21 / 1,111.34 ms with only aligned
convolution compilation (about 22,060 completion token/s, 1.435× faster).
Logprobs and loss were bit-exact in every arm. This has not established a BF16
win; its timing-only memory scope differs from the profiler peaks in the table.
The math SDPA control also preserved logprobs/loss exactly and selected
layer 0/3/23 gradients exactly, except conv weight relative L2 error 1.04e-5
from reduction order. FP32 master/gradient/Adam updates passed.
Input SHA-256: `70db852f422b237e762ac8685e151db92db9e996ef5630e4cef1188eace061c1`.
An accurate mixed-precision MFU needs operation counts; no new MFU is claimed.
The graph arm uses opt-in `RLFORGE_FP8_GRAPHS=1` and
`RLFORGE_FP8_GRAPH_MAX_MB=4096`; graph replay remains off by default.
A separate alternating test with convolution compilation enabled measured
1,113.04 / 1,116.82 ms without Linear graphs and 1,169.00 / 1,168.69 ms
with the 4-GiB graph cache. Graphs preserved logprobs/loss exactly but added
about 4.8% time in this configuration, supporting the default of off.

With the final FP32 tied-master serving fix and `FP8_ALIGN=vllm`, four
completions from one prompt had mean trainer/rollout log-ratios
`−0.001178, −0.000578, +0.000718, +0.000555`, all below 0.004 in absolute value.
Per-token mean absolute errors remained 0.055–0.093. This is a **prefill** gate;
it is not bit-exact. The autoregressive decode probe used the same unchanged
checkpoint, 128 generated tokens and one unique deterministic prompt/completion
(four repeated rows). Trainer teacher-forced prefill versus rollout decode had
mean log-ratio **+0.006154**, per-token mean absolute error **0.04432** and max
error **0.58437**. The `|mean log-ratio| < 0.004` gate **failed**. Broader
independent decode cases are preserved in the scale evidence JSON; long-run
learning is unvalidated.
An additional within-vLLM probe, with no HF trainer or weight reload, compared
teacher-forced prefill and decode on one fresh 128-token completion. The mean
prefill-minus-decode logprob was **−0.015703** (token absolute mean 0.039752),
also failing 0.004. This shows a serving-internal difference. Source inspection
found BF16-rounded Q/K in prefill versus FP32-normalized Q/K in recurrent decode,
plus different chunk/recurrent arithmetic; their individual contributions have
not been isolated. This probe generated a different completion, so its mean
cannot be subtracted from the preceding trainer/rollout result.

The live GPU reload probe passed all 13 checks, including FP32 tied masters,
updated quantized bytes/scales and preserved storage pointers. The fix upgrades
Qwen3.5's otherwise unquantized embedding to an FP32 master before checkpoint
loading and refreshes linked head caches when only the embedding weight loads.

Two-rank DDP passed two-microbatch FP32 accumulation, equal allreduced gradients
and nonzero FP32 Adam updates. The final aligned three-step trainer/rollout smoke
completed live sync and FP32 updates (selected step-3 max update 7.04e-7); it
establishes execution only. Early FA3 FP8 repeated gradients varied about
15–25% versus about 1% for BF16; the FP8 eager/graph replay control was exact
with math SDPA. This gradient sensitivity is still a quality boundary.
The remote environment lacked pytest; assertions were executed by saved direct
harnesses, including eight FP8 checks and twelve serving CPU checks.

Historical comparison corrections: [`FP8_HISTORY_2026-10-05.md`](reports/FP8_HISTORY_2026-10-05.md).

## Matched 4 + 4 scaling checks (2026-10-05)

The policy, frozen BF16 reference, data, reward and public hyperparameters are
matched to the BF16 control: G32, 1,024 real completions per optimizer step,
learning rate 2e-6, GSPO clip ±0.004 and KL coefficient 0.05. Throughput is the
sum of **trained completion tokens** divided by the sum of optimizer-step wall
seconds, using steps 4–20. See the compact table in
[`INFRA_HISTORY.md`](INFRA_HISTORY.md) and the source hashes and numerical checks
in [`FP8_SCALE_2026-10-05.json`](reports/FP8_SCALE_2026-10-05.json).

| Matched 4 trainer + 4 rollout path | Median step s | Total completion token/s | Rollout wait |
|---|---:|---:|---:|
| Historical BF16 | 35.57 | 53,691 | 6.4% |
| Current BF16 | 37.24 | 52,058 | 6.5% |
| FP8, quantization/cache fixes | 38.33 | 52,992 | 8.2% |
| FP8, compact backward + 128 GiB activation budget | 32.34 | 58,363 | 5.8% |
| FP8, detached view-base guards | 33.82 | 57,103 | 5.2% |
| **FP8, canonical backward strides (final)** | **32.23** | **61,268** | **4.1%** |

The final configuration completed 20 steps with 17 observations in the common
window: 33,484,351 trained completion tokens in 546.52 step seconds. Throughput
improved **17.7% over the current BF16 control and 14.1% over the historical
best**. All compact runs are preserved above. Sampled completion lengths and
judge outcomes differ; these short system measurements do not establish
statistical significance or long-run quality. All 320 final saved parameter
tensors were FP32 and the checked MLP master changed.

LR-zero log-ratio P90 remained 0.014004 versus 0.001551 for BF16, failing the
0.004 alignment gate; final low-side sequence clipping was 95.2%. Keep the
configuration experimental. There were zero compiler-limit fallbacks, CUDA OOMs
or tracebacks during the final run. Weight-sync pause consumed 8.9% of step wall
time; actual transfer consumed 0.4%.

The measured compact configuration adds:

```bash
RLFORGE_FP8_ALIGN_COMPILE=1 RLFORGE_FP8_ALIGN_POINTWISE=1 \
RLFORGE_FP8_ALIGN_BACKWARD=1 RLFORGE_SB_ACT_GB=128 \
RLFORGE_FP8_GRAPHS=0 RLFORGE_FP8_HEAD_GRAPH=0
```

The measured runs use trainer flags `--subbatch-tokens 131072 --sb-ckpt auto
--balance-rows on --dp-route on`, rollout TP1/DP4, queue 512, score concurrency
32 and inflight 960. Memory calibration is `RLFORGE_SB_MEM_FULL_KB=128`,
`RLFORGE_SB_MEM_CKPT_KB=4.7`, `RLFORGE_SB_MEM_BASE_KB=16`; frozen source hashes
and all input/settings hashes are recorded in the evidence JSON.

The activation budget was measured on the 0.8B H200 configuration; it is not a
universal safe memory limit. Compact backward supports first-order training.
Flattened backward inputs are detached from their original view bases to avoid
Dynamo guards on varying 2D/3D/4D shapes. Noncontiguous Q/K slices and upstream
gradients are made contiguous before compilation; already contiguous tensors
keep their storage. The detached-only follow-up still reached the eight-graph
limit on varying row strides; its judge failure fraction was 54.5%, so its
throughput is a system observation with different realized reward-service state.

Three concrete faults were repaired: variable token lengths created separate
Triton quantization variants; fused AdamW left version-keyed FP8 weights stale;
vLLM reused an AOT embedding kernel with the wrong input dtype. Runtime dimensions
passed 72 byte/scale/padding cases and six actual Linear forward/backward/update
cases. Ordinary and fused AdamW now refresh Linear, gate/up and head caches.
The identical-source run with a fresh, disabled AOT cache completed 20 steps.

Doubling the serving concurrency from 256 to 512 measured 15,881→16,038
completion token/s on one FP8 GPU. Fixed G64 trainer batches took about twice
the G32 time; G32 already fits in one subbatch. Increasing only the global
batch or token budget does not make those G32 GEMMs larger. G64 changes the
algorithm's group size and is not a matched throughput improvement.

Compact backward preserved fixed-batch logprobs and loss bit-for-bit. All 320
parameter gradients were finite FP32: aggregate cosine was 0.99143 versus
0.99152 for the unchanged FA3 repeat, so that full-model comparison contains
existing backward variation. Two-rank, two-microbatch `no_sync`/allreduce and
actual fused-Adam FP32 updates passed. This does not remove the prefill/decode
forward discrepancy or validate long-run learning.

A further fixed-token probe tested three prompts at 128 and 256 completions.
Both arms replayed exactly the same saved tokens with raw logprobs and unchanged
FP32 weights. Original vLLM prefill passed 2/6 sequence-mean gates; substituting
the trainer's installed FLA prefill passed 4/6, leaving means +0.004157 and
+0.010771 beyond the 0.004 threshold. The kernel wrapper matched the trainer's
output and final FP32 state bit-for-bit in four direct cases. The model candidate
was not promoted: shared prefill still uses chunk arithmetic while cached decode
uses a sequential recurrence, and tokenwise discrepancies remain.

True mixed-precision MFU remains unavailable. A separate fixed-G32 useful
projection/head GEMM estimate is 11.60% for BF16/TF32, 5.08% for repaired FP8
and 5.83% for compact128 (before the view-base guard repair), normalized by
their respective dense H200 peaks. It excludes attention/GDN
internal work, padding and recomputation; timing includes the BF16 reference.
These partial utilization estimates are not the historical dense TRL MFU.
