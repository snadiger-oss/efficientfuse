# efficientfuse (GreenFuse)

GreenFuse tests whether different ways of grouping GPU operations for compilation (TorchInductor), with or without CUDA Graph replay, reduce GPU energy while staying within 5% of the fastest runtime. The Python package and command are named `greenfuse`.

- **Results:** [results/haripreeth/README.md](results/haripreeth/README.md) (W1 + sin/tanh fan-out screen) and [results/bhargav/README.md](results/bhargav/README.md) (W1–W6, exp3–exp14).
- **History:** what was done in each version, and why, is in [PROJECT_VERSION_LOG.md](PROJECT_VERSION_LOG.md).

| Workload | Calculation |
|---|---|
| `w1` | `relu(x + bias) + residual` |
| `w2` | `gelu(x @ weight + bias)` |
| `w3` | `gelu(layer_norm(x + residual))` |
| `w4` | `h = x + bias; h - h.mean(-1, keepdim=True)` |
| `w5` | `softmax(gelu(x + bias), -1)` |
| `w6` | `gelu(layer_norm(x + residual) @ weight + bias)` |

## Repository structure

```
efficientfuse/
├── src/greenfuse/          the package
│   ├── cli.py              MAIN ENTRY POINT: the `greenfuse` command (inspect, calibrate, run, benchmark, select, report)
│   ├── workloads/          workloads W1–W6
│   ├── graph/, partition/  capture the op graph and enumerate groupings (p00 full, p01, p10, p11 separate)
│   ├── compiler/           compile each group with TorchInductor, read kernel evidence
│   ├── execution/          ordinary launch and CUDA Graph replay executors
│   ├── measurement/        runtime and NVML GPU energy measurement
│   ├── planner/            select within the 5% runtime bound, validate, cache decisions
│   └── naming.py           readable grouping names (full, first_pair, last_pair, separate)
├── configs/                experiment configs (YAML), one per experiment
├── scripts/
│   ├── check_environment.py, check_execution.py   setup checks
│   ├── export_experiment.py                       planner run -> readable summary.csv + plot
│   ├── compare_workloads.py, threshold_sensitivity.py   cross-experiment tables (Bhargav)
│   ├── screen_candidates.py, profile_kernels.py, quick_energy.py   manual GPU tools (Bhargav)
│   ├── exp3_screen_fanout.py                      sin/tanh fan-out screen (standalone)
│   ├── exp3_inspect_kernels.py, exp3_trace_report.py   fan-out screen kernel analysis
│   └── slurm/                                     Slurm jobs
├── tests/                  unit, integration, and GPU tests
├── results/                all measured results: haripreeth/ and bhargav/, one folder per person
├── docs/SETUP.md           detailed setup and command reference
└── PROJECT_VERSION_LOG.md  history: what changed in each version, findings, mistakes, reasons
```

## Reproduce

Needs a Linux machine with a whole NVIDIA GPU (results were measured on an A40) and gcc ≥ 9 (on Great Lakes: `module load gcc/13.2.0`).

**1. Install**

```bash
git clone https://github.com/HaripreethAvarur/efficientfuse.git && cd efficientfuse
conda create -n env_scalable python=3.11 -y && conda activate env_scalable
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
pip install -e ".[config,measurement,analysis,test]"
export CC=$(which gcc) CXX=$(which g++)
```

Bhargav's W1–W6 runs used PyTorch 2.10.0+cu128 with a venv instead; see [results/bhargav/README.md](results/bhargav/README.md#reproduce).

**2. Check the setup** (optional)

```bash
python scripts/check_environment.py --device cuda:0 --exclusive-gpu --output results/haripreeth/setup/environment.json
GREENFUSE_RUN_GPU_TESTS=1 GREENFUSE_EXCLUSIVE_GPU=1 python -m unittest discover -s tests
```

**3. W1, all groupings at all shapes** (covers Experiments 1 and 2)

```bash
greenfuse calibrate --config configs/all_groupings_w1_all_shapes.yaml --exclusive-gpu \
  --output-dir results/runs/all_groupings --cache-dir results/decisions

# Repeat for each shape; <run>/<id> are printed by calibrate
python scripts/export_experiment.py results/runs/all_groupings/<run>/w1_4096x4096/<id> \
  results/haripreeth/all_groupings_w1_all_shapes/4096x4096 --shape 4096x4096
```

Experiments 1 and 2 alone: use `configs/exp1_w1_small_last_pair_full.yaml` or `configs/exp2_w1_large_last_pair_full.yaml` in the same command.

**4. Any other experiment (W1–W6, exp3–exp14):** same `greenfuse calibrate` command with `configs/expN_*.yaml`.

**5. Sin/tanh fan-out screen** (where partial fusion wins; not the same as `configs/exp3_*`)

```bash
sbatch scripts/slurm/exp3_fanout_screen.sbatch     # set your own --account first, or run its python command directly
python scripts/exp3_inspect_kernels.py             # generated Triton code + per-kernel analysis
python scripts/exp3_trace_report.py --shape 8192x8192
```

Outputs go to `results/haripreeth/exp3_fanout_screen/`.
