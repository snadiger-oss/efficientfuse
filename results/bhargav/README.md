# GreenFuse results: workloads where full fusion might cost something (Bhargav)

Scope: three new workloads, each with all four groupings × both execution modes (8 compiled candidates), plus the eager baseline. They test the project hypothesis that partial grouping can beat full fusion, using workloads where full fusion has a plausible cost.

| Workload | Function | Operations (in order) | Why full fusion might lose |
|---|---|---|---|
| W2 | `gelu(x @ weight + bias)` | matmul, bias add, GELU | Fusing the epilogue into the GEMM needs a Triton GEMM instead of cuBLAS |
| W3 | `gelu(layer_norm(x + residual))` | residual add, LayerNorm, GELU | Fusing pointwise work into a reduction changes how the work is split |
| W4 | `h = x + bias; h - h.mean(-1, keepdim=True)` | bias add, row mean, subtract | `h` feeds two operations (fan-out), so full fusion may reload or recompute it |

All inputs are FP32 and contiguous, at 4096×4096 (W2: x [4096, 4096] @ weight [4096, 4096], so 4096³). Inference only, seed 0. LayerNorm has no affine parameters. GELU is the exact (erf) form.

## Candidate names

Same convention as `results/README.md`. Raw JSON keeps the internal IDs; summaries and plots use the readable names.

| Readable name | Internal ID | W2 groups | W3 groups | W4 groups |
|---|---|---|---|---|
| `compiled_separate_*` | `p11_*` | [matmul] [bias] [GELU] | [add] [LN] [GELU] | [add] [mean] [sub] |
| `compiled_first_pair_*` | `p01_*` | [matmul + bias] [GELU] | [add + LN] [GELU] | [add + mean] [sub] |
| `compiled_last_pair_*` | `p10_*` | [matmul] [bias + GELU] | [add] [LN + GELU] | [add] [mean + sub] |
| `compiled_full_*` | `p00_*` | [matmul + bias + GELU] | [add + LN + GELU] | [add + mean + sub] |
| `*_ordinary` / `*_replay` | `*_launch` / `*_graph` | ordinary execution / explicit CUDA Graph replay | | |
| `eager_baseline` | `eager_launch` | raw PyTorch eager; reference and baseline, not selectable | | |

## Folder layout

```
results/bhargav/
├── README.md                           this file
├── summary_all_experiments.csv         every candidate, experiment and pass in one table
├── energy_by_workload.png              energy of each grouping relative to the best option
├── setup/                              environment.json, packages, commit, nvidia-smi,
│                                       execution_checks_*.json (correctness + kernel names per candidate),
│                                       w2_autotune_probe.txt, profiler traces
├── exp3_w2_linear_gelu_cublas/         4096x4096/ {summary.csv, runtime_vs_energy.png, raw/}, logs/
├── exp4_w2_linear_gelu_triton_gemm/    same layout
├── exp5_w3_layernorm_reduction/        same layout
├── exp6_w4_fan_out/                    same layout
└── w1_32x256_measure_only/             earlier W1 separate/first_pair run (measurement only, no selection)
```

Configs: `configs/exp3_w2_linear_gelu_cublas.yaml`, `exp4_w2_linear_gelu_triton_gemm.yaml`, `exp5_w3_layernorm_reduction.yaml`, `exp6_w4_fan_out.yaml`.

## Hardware and software

- **Hardware:** NVIDIA A40 (48 GB, 6 MB L2), driver 595.71, Great Lakes node with a whole reserved GPU. The NVML energy counter was available.
- **Software:** PyTorch **2.10.0+cu128**, Triton 3.6.0, Python 3.12.1, system gcc 8.5.
- **Numerics:** TF32 off (`torch.backends.cuda.matmul.allow_tf32 = False`, `float32_matmul_precision = highest`).
- **Tolerances:** correctness checked at rtol = atol = 1e-5 on two input sets. The largest observed error was 1.4e-6.
- **Measurement:** 5 rounds, 3 s windows, separate timing and energy passes, shuffled order, then a fresh validation repeat. Same policy as the other pair.

**Version note:** the other pair used PyTorch 2.15.0+cu132. Comparisons within this folder are consistent; **do not pool absolute numbers with `results/exp1_*` or `results/exp2_*`.** At W1 32×256, for example, our eager baseline measured 53 µs and theirs 86 µs on the same GPU model.

## Results (fresh validation pass, ordinary execution, µs / mJ per call)

| Grouping | W2 cuBLAS (exp3) | W2 Triton GEMM (exp4) | W3 LayerNorm (exp5) | W4 fan-out (exp6) |
|---|---|---|---|---|
| full | 6371 / 1873 | 6910 / 2093 | **393 / 93.6** | **252 / 54.4** |
| first_pair | 6388 / 1923 | 7047 / 2131 | 627 / 140.9 | 477 / 102.4 |
| last_pair | **6370 / 1869** | 7061 / 2090 | 603 / 135.7 | 486 / 103.5 |
| separate | 6596 / 1967 | 7256 / 2197 | 841 / 187.5 | 585 / 126.7 |
| eager baseline | 6602 / 1976 | 6598 / 1987 | 957 / 213.6 | 611 / 139.0 |
| **Verdict** | inconclusive | inconclusive | **validated: full_ordinary** | **validated: full_ordinary** |

Kernels per grouping (from `setup/execution_checks_*.json`):

| Grouping | W2 cuBLAS | W2 Triton GEMM | W3 | W4 |
|---|---|---|---|---|
| full | cuBLAS GEMM + `fused_add_gelu` | **one** `triton_tem_fused_add_gelu_mm` | one `red_fused_add_gelu_native_layer_norm` | one `red_fused_add_mean_sub` |
| last_pair | cuBLAS GEMM + `fused_add_gelu` | Triton GEMM + `fused_add_gelu` | add + `red_fused_gelu_native_layer_norm` | add + `red_fused_mean_sub` |
| first_pair | cuBLAS GEMM (addmm) + `gelu` | `triton_tem_fused_addmm` + `gelu` | `red_fused_add_native_layer_norm` + `gelu` | `red_fused_add_mean` + `sub` |
| separate | cuBLAS GEMM + add + gelu | Triton GEMM + add + gelu | add + layer_norm + gelu | add + mean + sub |

## What the results show

**W2 linear + GELU: partial fusion beats full fusion when full fusion means changing the GEMM.**

| Comparison (ordinary execution) | Time | Energy |
|---|---|---|
| cuBLAS GEMM + separate bias/GELU kernel (exp3, `last_pair`) | 6370 µs | 1869 mJ |
| One Triton GEMM with bias/GELU fused in (exp4, `full`) | 6910 µs | 2093 mJ |
| **Difference** | **8% faster** | **11% less energy** |

- **This is a comparison across two runs.** The eager baselines in the two runs agree within 0.1% (6602 vs 6598 µs), so they are comparable. The planner itself cannot make this choice, because one run uses a single GEMM policy for all groups.
- **Inductor already makes this choice by default.** With autotuning on, it benchmarks cuBLAS at 5.78 ms against the best Triton GEMM at 6.38 ms and keeps cuBLAS for every grouping (`setup/w2_autotune_probe.txt`). As a result, `full` and `last_pair` in exp3 compile to identical kernels and tie, which is why exp3 is inconclusive.
- **Exp4's verdict is also a tie.** Its validation pass put last_pair and full within 0.2% on energy, so it is inconclusive too.
- **The GEMM dominates W2.** It runs at about 21.6 TFLOPS, 58% of the A40's FP32 peak, so grouping changes only the remaining ~3–5%.

**W3 LayerNorm and W4 fan-out: full fusion wins clearly, and the possible costs did not appear.**

- **W3 `full` is the best option** (validated and cached). Compared with the best partial grouping it is 35% faster and uses 31% less energy. Compared with eager it is 2.4× faster and uses 2.3× less energy.
- **W4 `full` is the best option** (validated and cached). Compared with the best partial grouping it is 47% faster and uses 47% less energy.
- **W4 shows no reload or recompute.** `full` moves 128 MB per call in 252 µs, about 533 GB/s (77% of peak). Re-reading `h` would need about 763 GB/s, which is above the A40's peak. Inductor's fused reduction keeps each 4096-wide row on chip, so splitting the groups only adds a full write and re-read of `h`.

**Overall:** on these workloads and settings, the only case where partial grouping won is W2 GEMM policy (cuBLAS + separate epilogue vs a fused Triton GEMM). In that case Inductor's default already makes the better choice. These conclusions are limited to 4096×4096 FP32 on the A40.

## Known issues

**Replay numbers include input copies.** This is the same issue described in `results/README.md`. In every workload, replay minus ordinary equals the cost of copying the inputs each call:

| Workload | Replay overhead | Copy cost |
|---|---|---|
| W2 | +459 µs | 2 × 64 MB inputs |
| W3 | +480 to +498 µs | 2 × 64 MB |
| W4 | +245 µs | 1 × 64 MB |

So replay looks worse than it is. Do not read "ordinary beats replay" as a finding until the team decides how to measure refresh cost.

**Kernel counts during calibration.** PyTorch 2.10 keeps its profiler's GPU tracing (CUPTI) alive once CUDA Graphs are used, and later profiler sessions drop GPU kernel records. Calibration therefore:
- verifies the execution mode in every preparation from CPU-side launch records;
- takes ordinary-mode kernel counts from launch API calls;
- leaves replay-mode counts as unknown when they can't be seen.

Authoritative kernel names and counts for every candidate come from fresh-process checks in `setup/execution_checks_*.json`.

## Reproduce

On a GPU node with the project venv:

```bash
module load python/3.12.1
source .venv/bin/activate
export C_INCLUDE_PATH=/sw/pkgs/arc/python/3.12.1/include/python3.12   # Great Lakes: Triton/Inductor need Python.h
export CPLUS_INCLUDE_PATH=$C_INCLUDE_PATH

# Correctness + kernel names for all 8 candidates
python scripts/check_execution.py --device cuda:0 --workload w2 --m 4096 --n 4096 \
  --output results/bhargav/setup/execution_checks_w2_4096.json
python scripts/check_execution.py --device cuda:0 --workload w2 --gemm-backend triton --m 4096 --n 4096 \
  --output results/bhargav/setup/execution_checks_w2_triton_4096.json
# (w3, w4 likewise)

# Search, select, validate, cache (one per experiment)
greenfuse calibrate --config configs/exp5_w3_layernorm_reduction.yaml --exclusive-gpu \
  --output-dir results/bhargav/runs/exp5_w3_layernorm_reduction --cache-dir results/decisions

# Readable folder + cross-workload table/figure
python scripts/export_experiment.py results/bhargav/runs/exp5_w3_layernorm_reduction/<run>/w3_4096x4096/<id> \
  results/bhargav/exp5_w3_layernorm_reduction/4096x4096 --shape 4096x4096 --label "W3 add+LayerNorm+GELU"
python scripts/compare_workloads.py
```

## Code changes behind these runs

1. **New workloads** W2–W4 in `workloads/definitions.py`, `specs.py` and `registry.py`. W3 is built per row width, because LayerNorm needs a fixed `normalized_shape`.
2. **New supported operations:** matmul, GELU, LayerNorm, mean and subtract. Tuple constants are now allowed.
3. **Fan-out support:** an operation may reuse earlier results as long as it still consumes the previous one. Groups can now pass several values onward (`GroupBindings.output_ids`).
4. **`gemm_backend` config key** (`default` / `autotune` / `triton`) and a `--gemm-backend` script flag. Autotuning is restricted to GEMMs, so Inductor never adds its own CUDA Graphs.
5. **TF32 settings** are recorded in the environment report and the planner identity.
6. **Profiler fixes** (see Known issues), and a `--workload` option for `check_execution.py` and `measure_workload.py`.
7. **`--partitions`** now also accepts names: `separate`, `first_pair`, `last_pair`, `full`.
8. **New scripts:** `scripts/compare_workloads.py` for the cross-workload table and figure, and a `--label` option for `export_experiment.py`.
