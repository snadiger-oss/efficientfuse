# GreenFuse results: W1–W6 grouping and CUDA Graph experiments (Bhargav)

Scope: experiments 3–14. Part 1 (exp3–6) tests large shapes where full fusion might have a cost. Part 2 (exp7–14) tests small and decode-style shapes with replay reading inputs already on the GPU, and adds a policy where fusing an epilogue into a matrix multiply forces a Triton GEMM. Every experiment measures all groupings in ordinary and replay modes, plus the eager baseline.

| Workload | Function | Operations (in order) | Why full fusion might lose |
|---|---|---|---|
| W2 | `gelu(x @ weight + bias)` | matmul, bias add, GELU | Fusing the epilogue into the GEMM needs a Triton GEMM instead of cuBLAS |
| W3 | `gelu(layer_norm(x + residual))` | residual add, LayerNorm, GELU | Fusing pointwise work into a reduction changes how the work is split |
| W4 | `h = x + bias; h - h.mean(-1, keepdim=True)` | bias add, row mean, subtract | `h` feeds two operations (fan-out), so full fusion may reload or recompute it |
| W5 | `softmax(gelu(x + bias), -1)` | bias add, GELU, softmax | A multi-pass reduction may recompute the pointwise work |
| W6 | `gelu(layer_norm(x + residual) @ weight + bias)` | residual add, LayerNorm, matmul, bias add, GELU | Five operations, so the best plan may fuse some regions and cut others (16 groupings × 2 modes = 32 candidates) |
| W1 | `relu(x + bias) + residual` | bias add, ReLU, residual add | Control: elementwise only, used here at a launch-bound shape |

All inputs are FP32 and contiguous, at 4096×4096 (W2: x [4096, 4096] @ weight [4096, 4096], so 4096³). Inference only, seed 0. LayerNorm has no affine parameters. GELU is the exact (erf) form.

## Candidate names

Same convention as `results/haripreeth/README.md`. Raw JSON keeps the internal IDs; summaries and plots use the readable names.

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
├── overnight_summary.md                every candidate for exp7–14: runtime and energy
├── summary_all_experiments.csv         exp3–6 in one table
├── energy_by_workload.png              exp3–6 energy of each grouping relative to the best option
├── threshold_sensitivity.csv           selections at 1.05×, 1.10× and 1.20× for every run
├── setup/                              environment, packages, commit, nvidia-smi, execution_checks_*.json,
│                                       w2_autotune_probe.txt
├── screening/                          runtime screens used to choose exp7–14
├── kernel_profiles/                    per-kernel GPU time for selected plans, one fresh process per plan
├── quick_energy/                       W1 single-window runtime and energy, not a calibration
├── logs/overnight.log                  unattended run log for exp7–14
├── exp3_… to exp14_…/                  <shape>/{summary.csv, runtime_vs_energy.png, raw/}, logs/
└── w1_32x256_measure_only/             earlier W1 separate/first_pair run (measurement only, no selection)
```

Configs are in `configs/exp3_*.yaml` through `configs/exp14_*.yaml`. Profiler trace folders were removed from the repository; `trace_path` values inside `setup/execution_checks_*.json` are historical.

## Hardware and software

- **Hardware:** NVIDIA A40 (48 GB, 6 MB L2), driver 595.71, Great Lakes node with a whole reserved GPU. The NVML energy counter was available.
- **Software:** PyTorch **2.10.0+cu128**, Triton 3.6.0, Python 3.12.1, system gcc 8.5.
- **Numerics:** TF32 off (`torch.backends.cuda.matmul.allow_tf32 = False`, `float32_matmul_precision = highest`).
- **Tolerances:** correctness checked at rtol = atol = 1e-5 on two input sets. The largest observed error was 1.4e-6.
- **Measurement:** 5 rounds, 3 s windows, separate timing and energy passes, shuffled order, then a fresh validation repeat. Same policy as the other pair.

**Version note:** the other pair used PyTorch 2.15.0+cu132. Comparisons within this folder are consistent; **do not pool absolute numbers with `results/haripreeth/`.** At W1 32×256, for example, our eager baseline measured 53 µs and theirs 86 µs on the same GPU model.

## Part 1: exp3–6 at 4096×4096 (fresh validation pass, ordinary execution, µs / mJ per call)

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

## What Part 1 shows

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

**Part 1 overall:** on these workloads and settings, the only case where partial grouping won is W2 GEMM policy (cuBLAS + separate epilogue vs a fused Triton GEMM). In that case Inductor's default already makes the better choice. These conclusions are limited to 4096×4096 FP32 on the A40.

## Part 2: exp7–14 with GPU-resident replay inputs

These runs use `replay_input_refresh: on_change`: replay copies inputs only when they are new tensors or were modified in place. Runtime and energy are per call from the fresh validation pass. The table compares full fusion + replay with the lowest-energy partial grouping + replay. Negative percentages favor partial grouping. **Result** uses a ±2% energy band for a tie; **Planner verdict** is the selection and repeat outcome across all candidates.

| Experiment | Workload, shape | GEMM policy | Full + replay | Lowest-energy partial + replay | Time | Energy | Result | Planner verdict |
|---|---|---|---|---|---|---|---|---|
| exp7 | W1 256×1024 | `default` | 37.3 µs / 3.11 mJ | `first_pair`: 37.3 µs / 3.28 mJ | +0.1% | +5.3% | Full | validated |
| exp8 | W3 32×1024 | `default` | 34.9 µs / 2.80 mJ | `first_pair`: 34.9 µs / 2.79 mJ | 0.0% | −0.2% | Tie | inconclusive |
| exp9 | W4 256×4096 | `default` | 33.7 µs / 4.76 mJ | `last_pair`: 33.8 µs / 6.15 mJ | +0.3% | +29.1% | Full | validated |
| exp10 | W5 32×1024 | `default` | 33.3 µs / 2.68 mJ | `last_pair`: 34.0 µs / 2.74 mJ | +1.9% | +2.1% | Full | validated |
| exp11 | W2 128×4096 | `default` | 260.2 µs / 78.08 mJ | `last_pair`: 261.0 µs / 77.90 mJ | +0.3% | −0.2% | Tie | inconclusive |
| exp12 | W2 16×4096 | `fuse_epilogue` | 115.7 µs / 34.53 mJ | `first_pair`: 118.2 µs / 34.75 mJ | +2.2% | +0.6% | Tie | inconclusive |
| exp12 | W2 64×4096 | `fuse_epilogue` | 175.9 µs / 52.06 mJ | `last_pair`: 144.8 µs / 42.90 mJ | −17.7% | −17.6% | Partial | inconclusive |
| exp12 | W2 128×4096 | `fuse_epilogue` | 300.9 µs / 90.33 mJ | `last_pair`: 262.8 µs / 77.98 mJ | −12.7% | −13.7% | Partial | validated |
| exp13 | W6 64×4096 | `fuse_epilogue` | 180.5 µs / 54.15 mJ | `add+ln_mm_add_gelu`: 154.1 µs / 45.42 mJ | −14.7% | −16.1% | Partial | inconclusive |
| exp14 | W6 512×4096 | `fuse_epilogue` | 1100.1 µs / 325.07 mJ | `add+ln_mm_add+gelu`: 954.5 µs / 286.44 mJ | −13.2% | −11.9% | Partial | inconclusive |

`fuse_epilogue` is an experimental compiler policy: a group that fuses a matmul with other operations uses a Triton GEMM, while a matmul-only group keeps cuBLAS. An `inconclusive` verdict means the planner's winner changed between passes. In W2 64×4096 and W6 64×4096 both winners were partial plans; in W6 512×4096 both passes chose the same grouping but different execution modes.

Findings, limited to these settings:

- **Default compiler policy:** partial + replay did not save energy. It tied or used more energy than full + replay.
- **`fuse_epilogue`:** at W2 64 and 128 rows and W6 64 and 512 rows, partial + replay was 13–18% faster and used 12–18% less GPU energy than full + replay. Only W2 128×4096 has a validated planner decision.
- **Why:** kernel profiles show the fused Triton matrix multiply taking longer than cuBLAS: 154.4 vs 130.0 µs (W2 64), 264.0 vs 227.0 µs (W2 128), 154.8 vs 130.5 µs (W6 64), 950.0 vs 780.8 µs (W6 512). The extra epilogue kernels are small. At 64 rows, the measured end-to-end gap is larger than the kernel-time gap and is not fully explained.
- **Threshold sensitivity:** recomputing the rule at 1.10× and 1.20× changed no selected candidate in the 14 exported runs. Those allowances were added after seeing data and are secondary analysis.

`quick_energy/` contains one 2-second ordinary-execution window per W1 plan. Use it as indicative data, not as a calibration.

## Known issues

**Replay numbers in Part 1 (exp3–6) include input copies.** This is the same issue described in `results/haripreeth/README.md`; it does not apply to exp7–14. In every Part 1 workload, replay minus ordinary equals the cost of copying the inputs each call:

| Workload | Replay overhead | Copy cost |
|---|---|---|
| W2 | +459 µs | 2 × 64 MB inputs |
| W3 | +480 to +498 µs | 2 × 64 MB |
| W4 | +245 µs | 1 × 64 MB |

So replay looks worse than it is in Part 1. Do not read those rows as "ordinary beats replay".

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
9. **Replay input policy:** `replay_input_refresh` keeps the original copy-every-call behavior by default; `on_change` measures replay with resident inputs.
10. **W5, W6 and `fuse_epilogue`:** added for exp7–14, with `scripts/screen_candidates.py`, `profile_kernels.py`, `quick_energy.py`, `threshold_sensitivity.py` and `scripts/slurm/overnight_bhargav.sh`.
