# GreenFuse results: last_pair and full groupings

Scope: this half of the team's search space. The other half (separate, first_pair) is run separately.

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
results/
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
└── decisions/                         planner cache (hash file names are required by the code)
```

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

## Known issue: replay numbers include input copies

Replay mode copies every input into its own buffers on every call; see `src/greenfuse/execution/graphs.py`, `run()`. That adds 4 tensor reads and writes per call that ordinary mode does not do, and the numbers match it:

- full_replay: 7 vs 3 tensor passes, predicted ≈ 830 µs, measured 840 µs at 4096².
- last_pair_replay: 9 vs 5 tensor passes, predicted ≈ 1066 µs, measured 1079 µs at 4096².

**Do not read "replay is slower at large shapes" as a finding.** Ordinary-vs-ordinary and ordinary-vs-eager comparisons are unaffected. The team needs to decide whether to measure replay without the copy and report the refresh cost separately, which is what the project spec asks for.

## Reproduce

Run on a GPU node:

```bash
cd efficientfuse
module load gcc/13.2.0                     # system gcc 8.5 breaks TorchInductor
export CC=$(which gcc) CXX=$(which g++)
conda activate env_scalable

python scripts/check_environment.py --device cuda:0 --exclusive-gpu --output results/setup/environment.json
python scripts/check_execution.py --device cuda:0 --partitions p10 p00 --compiler-artifacts \
  --output results/setup/execution_checks_last_pair_and_full.json
GREENFUSE_RUN_GPU_TESTS=1 GREENFUSE_EXCLUSIVE_GPU=1 python -m unittest discover -s tests

greenfuse calibrate --config configs/exp1_w1_small_last_pair_full.yaml --exclusive-gpu \
  --output-dir results/runs/exp1 --cache-dir results/decisions
greenfuse calibrate --config configs/exp2_w1_large_last_pair_full.yaml --exclusive-gpu \
  --output-dir results/runs/exp2 --cache-dir results/decisions

# Turn a planner run folder into a readable experiment folder
python scripts/export_experiment.py results/runs/exp2/<run>/w1_4096x4096/<id> \
  results/exp2_w1_large_shapes/4096x4096 --shape 4096x4096

# Use a cached decision (only for shapes whose calibration reported "validated")
greenfuse run --config configs/exp2_w1_large_last_pair_full.yaml --exclusive-gpu \
  --cache-dir results/decisions --output-dir results/runs/exp2_cached
```

## Code changes made for these runs (uncommitted)

1. `partitions:` config key and `--partitions` script flag to run a subset of groupings.
2. `src/greenfuse/compiler/inspect.py`: PyTorch 2.15 writes the invocation marker on both the CPU and GPU timelines; the parser now counts only the CPU marker.
3. `src/greenfuse/measurement/environment.py`: the torch version now comes from `torch.__version__`. This matches the compiler record, so selection no longer rejects valid data.
4. `scripts/export_experiment.py`: produces the readable summaries and plots in this folder.

Version note: the team guide pins torch 2.10.0+cu128 and this run used 2.15.0+cu132. Use the same version before pooling numbers with the other pair. Planner cache keys include the torch version, so cached decisions from the two versions will not match.

Next: Experiment 3, a workload where full fusion has a real cost (for example linear + GELU), is on hold.
