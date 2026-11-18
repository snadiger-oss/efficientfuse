# GreenFuse results (Haripreeth)

Experiments 1–2 cover the last_pair and full groupings; the all-groupings run repeats them with all four groupings; Experiment 3 uses a second workload where partial fusion can win. How to rerun everything: see the [top-level README](../../README.md#reproduce).

**Naming:** "Experiment 3" in this file is the sin/tanh fan-out screen (`exp3_fanout_screen/`). Bhargav's exp3–exp14 (W1–W6) are a separate series in [../bhargav/README.md](../bhargav/README.md); his exp3 is W2 linear + GELU.

Workload W1: `torch.relu(x + bias) + residual`, FP32, contiguous, inference only, seed 0. `x` and `residual` are `[M, N]`; `bias` is `[N]`.

## Candidate names

Raw JSON files keep the planner's internal IDs because the IDs are part of their fingerprints. Every summary, plot, and README uses the readable names.

| Readable name | Internal ID | Compilation groups | Execution |
|---|---|---|---|
| `compiled_last_pair_ordinary` | `p10_launch` | [bias add] [ReLU + residual add] | ordinary |
| `compiled_last_pair_replay` | `p10_graph` | [bias add] [ReLU + residual add] | CUDA Graph replay |
| `compiled_full_ordinary` | `p00_launch` | [bias add + ReLU + residual add] | ordinary |
| `compiled_full_replay` | `p00_graph` | [bias add + ReLU + residual add] | CUDA Graph replay |
| `eager_baseline` | `eager_launch` | raw PyTorch eager | correctness reference and baseline; not selectable |

Teammate groupings, not run here: `separate` = `p11`, `first_pair` = `p01`.

## Folder layout

```
results/haripreeth/
├── README.md                          this file
├── setup/                             environment and correctness evidence
│   ├── environment.json               GPU, NVML energy counter, versions
│   ├── execution_checks_last_pair_and_full.json   correctness + observed kernel counts
│   ├── compiler_artifacts_last_pair_and_full/     Inductor debug code and profiler traces
│   ├── packages.txt, commit.txt, nvidia-smi.txt
├── exp1_w1_small_shape/
│   ├── 32x256/                        summary.csv, runtime_vs_energy.png, raw/
│   └── logs/calibrate.log
├── exp2_w1_large_shapes/
│   ├── summary_all_shapes.csv         every candidate, shape and pass in one table
│   ├── energy_by_shape.png
│   ├── 1024x1024/  4096x4096/  8192x8192/   summary.csv, runtime_vs_energy.png, raw/
│   └── logs/                          attempt 1 (interrupted during 4096) and attempt 2
├── all_groupings_w1_all_shapes/       all 4 groupings x 2 modes, 4 shapes (same layout as exp2)
└── exp3_fanout_screen/
    ├── summary.csv                    every shape, k and candidate
    ├── raw/                           per-(shape, k) trial data
    ├── kernel_analysis/               generated Triton code + per-kernel stats
    ├── kernel_trace/                  KERNEL_TRACE_<shape>.md explanations, Chrome traces
    └── progress.log, slurm-*.log, environment.json
```

The planner decision cache (`results/decisions/`) is written by `greenfuse calibrate` but not committed.

Each shape folder contains:

- `summary.csv`: readable names, initial and validation passes, runtime and energy medians with min/max, and the selected flag.
- `runtime_vs_energy.png`: initial search results, with the fastest-candidate line, the 1.05× cutoff, and the selected candidate circled.
- `raw/`: unchanged planner output. It holds `initial.json` (search trials), `validation.json` (fresh trials), `decision.json` (selection evidence and verdict), the CSVs, and per-trial kernel evidence.

## Hardware and software

NVIDIA A40 (48 GB, 6 MB L2), driver 595.71, Great Lakes Slurm job on a whole reserved GPU. PyTorch 2.15.0+cu132, Triton 3.9, gcc 13.2. The NVML cumulative energy counter was available.

Metrics:

- **Runtime:** synchronized wall-clock time per call.
- **Energy:** GPU-device energy per call, not whole-system energy.

Each candidate ran 5 rounds of 3 s windows, with timing and energy in separate passes and candidate order shuffled.

## Experiment 1: small shape (32×256), inconclusive

Config: `configs/exp1_w1_small_last_pair_full.yaml`

| Candidate | Initial: µs / mJ per call | Validation: µs / mJ per call |
|---|---|---|
| compiled_last_pair_ordinary | 202.4 / 14.99 | 206.9 / 15.36 |
| compiled_last_pair_replay | 97.3 / 7.37 | 97.6 / 7.51 |
| compiled_full_ordinary | 122.2 / 8.86 | 121.8 / 9.05 |
| compiled_full_replay | 96.6 / 7.48 | 97.3 / 7.41 |
| eager_baseline | 86.5 / 6.38 | 86.3 / 6.52 |

**Verdict: inconclusive. No decision was cached.**

- The initial search picked last_pair_replay; the fresh validation pass picked full_replay. The two are about 1.5% apart.
- At this size the GPU work takes about 0.1 µs. Each call is dominated by CPU-side overhead, so only replay vs ordinary matters here.

## Experiment 2: large shapes, validated

Config: `configs/exp2_w1_large_last_pair_full.yaml`

| Shape (MB per tensor) | Selected (validated, cached) | Why |
|---|---|---|
| 1024×1024 (4 MB, fits in L2) | `compiled_full_replay` | Still overhead-bound. full_ordinary used less energy but was 27% slower than the fastest. |
| 4096×4096 (64 MB) | `compiled_full_ordinary` | Fastest and lowest energy |
| 8192×8192 (256 MB) | `compiled_full_ordinary` | Fastest and lowest energy |

Fresh validation pass, µs / mJ per call:

| Candidate | 1024×1024 | 4096×4096 | 8192×8192 |
|---|---|---|---|
| compiled_full_ordinary | 124.2 / 12.66 | **355.9 / 78.5** | **1392 / 317** |
| compiled_full_replay | **99.1 / 14.26** | 839.7 / 187.5 | 3309 / 740 |
| compiled_last_pair_ordinary | 209.4 / 20.48 | 592.5 / 132.6 | 2333 / 525 |
| compiled_last_pair_replay | 100.3 / 16.06 | 1078.8 / 234.1 | 4251 / 949 |
| eager_baseline | 88.1 / 12.19 | 846.7 / 191.7 | 3354 / 763 |

What the large shapes show:

- **Full beats last_pair by about 40% in time and energy.** Full moves 3 tensors per call (reads plus writes) and last_pair moves 5. The predicted ratio is 0.60; the measured ratio is 0.59–0.60.
- **Full uses about 59% less energy than eager.**
- **The kernels are memory-bound.** compiled_full_ordinary at 4096² reaches about 566 GB/s, roughly 81% of the A40's peak bandwidth.
- **Run-to-run variation is under 1%.**

## All groupings, all shapes (validated)

Config: `configs/all_groupings_w1_all_shapes.yaml`

- **What ran:** all 8 compiled candidates plus eager, measured under identical conditions in one run on the A40.
- **Folder:** `all_groupings_w1_all_shapes/`, with one subfolder per shape plus `summary_all_shapes.csv`, `energy_by_shape.png`, and `logs/`.
- **Names:** `first_pair` = `p01`, `separate` = `p11`.

| Shape | Selected (validated, cached) |
|---|---|
| 32×256 | `compiled_full_replay` |
| 1024×1024 | `compiled_full_replay` |
| 4096×4096 | `compiled_full_ordinary` |
| 8192×8192 | `compiled_full_ordinary` |

Fresh validation pass, µs / mJ per call:

| Candidate | 32×256 | 1024×1024 | 4096×4096 | 8192×8192 |
|---|---|---|---|---|
| compiled_full_ordinary | 120.6 / 9.14 | 123.3 / 12.36 | **354.6 / 78.50** | **1393 / 317.4** |
| compiled_full_replay | **95.9 / 7.31** | **96.9 / 14.23** | 839.2 / 187.2 | 3309 / 743.0 |
| compiled_first_pair_ordinary | 205.7 / 15.11 | 207.0 / 20.55 | 590.6 / 127.1 | 2332 / 504.1 |
| compiled_first_pair_replay | 96.8 / 7.36 | 98.2 / 16.08 | 1077.6 / 236.4 | 4252 / 942.4 |
| compiled_last_pair_ordinary | 204.7 / 15.51 | 207.9 / 20.48 | 590.7 / 129.2 | 2332 / 512.8 |
| compiled_last_pair_replay | 96.8 / 7.46 | 98.1 / 16.15 | 1077.6 / 238.0 | 4250 / 950.7 |
| compiled_separate_ordinary | 285.8 / 21.59 | 286.5 / 27.68 | 827.8 / 177.6 | 3275 / 721.7 |
| compiled_separate_replay | 98.6 / 7.61 | 99.4 / 17.09 | 1311.3 / 282.4 | 5190 / 1148 |
| eager_baseline | 84.9 / 6.44 | 86.1 / 12.14 | 847.6 / 191.1 | 3354 / 764.3 |

What the results show:

- **Large shapes (4096², 8192²) track memory traffic exactly.** Each call moves separate : pair : full = 7 : 5 : 3 tensors (reads plus writes), and ordinary-mode runtime follows suit. At 4096² the measured ratio is 2.33 : 1.67 : 1, against a prediction of 2.33 : 1.67 : 1.
  - Energy follows the same ordering.
  - first_pair and last_pair are indistinguishable because they move the same amount of data.
- **Small shapes (32×256, 1024²) are dominated by per-call overhead.**
  - Replay makes every grouping roughly the same cost.
  - In ordinary mode, cost grows with the number of compiled groups.
  - Eager is still the fastest option at these sizes.
- **Full fusion wins at every shape.** No partial grouping beat it. This was expected for a pure elementwise chain: fusion only removes traffic here. A case where partial fusion could win needs a workload where full fusion has a cost, which is Experiment 3.

## Known issue: replay numbers include input copies

Replay mode copies every input into its own buffers on every call; see `src/greenfuse/execution/graphs.py`, `run()`. That adds 4 tensor reads and writes per call that ordinary mode does not do, and the numbers match it:

- full_replay: 7 vs 3 tensor passes, predicted ≈ 830 µs, measured 840 µs at 4096².
- last_pair_replay: 9 vs 5 tensor passes, predicted ≈ 1066 µs, measured 1079 µs at 4096².

**Do not read "replay is slower at large shapes" as a finding.** Bhargav later added `replay_input_refresh: on_change` (copy only new or modified inputs) and used it for his exp7–14; the default is still `always`, and every run in this file used `always`. Ordinary-vs-ordinary and ordinary-vs-eager comparisons are unaffected. The team needs to decide whether to measure replay without the copy and report the refresh cost separately, which is what the project spec asks for.

## Rerunning

Steps are in the [top-level README](../../README.md#reproduce). On Great Lakes, load `gcc/13.2.0` first (system gcc 8.5 breaks TorchInductor).

Version note: the team guide pins torch 2.10.0+cu128 and this run used 2.15.0+cu132. Use the same version before pooling numbers with the other pair. Planner cache keys include the torch version, so cached decisions from the two versions will not match.

## Experiment 3 (sin/tanh fan-out screen): partial fusion wins

Script: `scripts/exp3_screen_fanout.py` (Slurm: `scripts/slurm/exp3_fanout_screen.sbatch`). Folder: `exp3_fanout_screen/`.

Workload: `y = E_k(x)` (k alternating sin/tanh ops), then `out = y * y.sum(-1, keepdim=True)`. Rows are too long for a persistent reduction, so the fully fused kernel walks each row twice and recomputes `y` in the second pass. Splitting after `y` stores it once (one extra write + read) instead.

Candidates: `compiled_full`, `compiled_split_after_y` = [E_k] [sum + multiply], `compiled_separate`, and eager. Ordinary execution only; 5 rounds of 2 s windows, shuffled order.

split_after_y relative to full (time / energy; below 1 means the split wins):

| k | 8192×8192 | 2048×32768 |
|---|---|---|
| 2 | 1.93 / 1.55 | 1.83 / 1.53 |
| 4 | 1.97 / 1.72 | 1.44 / 1.30 |
| 8 | 1.19 / 1.19 | 1.15 / 1.15 |
| 12 | **0.88 / 0.88** | **0.86 / 0.85** |
| 16 | **0.79 / 0.83** | **0.77 / 0.77** |
| 24 | **0.68 / 0.70** | **0.66 / 0.70** |
| 32 | 1.22 / 1.23 | 1.16 / 1.15 |

- **For 12 ≤ k ≤ 24 the partial grouping beats full fusion by 12–34% in time and 12–30% in energy** at both shapes.
- Why (from the generated code in `kernel_trace/`): full fusion evaluates the sin/tanh chain twice per element (2k ops); the split evaluates it once.
- At k = 32 Inductor changes strategy and the fused kernel stores `y` instead of recomputing it, so full wins again.
- This is a standalone screen, not yet run through the planner's separate timing/energy passes and validation.
